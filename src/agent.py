"""Loop principal do agente. State machine com handlers por estado.

Filosofia: capture → detect_state → handler. Cada handler executa UMA micro-ação
via gamepad e retorna ao loop, que recaptura. Erros não acumulam — cada passo
se auto-corrige no próximo screenshot.

O que mudou com o sensor determinístico: estado, cartas, cursor e navegação no
mapa vêm de `src.vision`, não do VLM. O modelo decide (qual carta jogar, qual
recompensa pegar) e o código valida antes de executar.
"""

from __future__ import annotations

import argparse
import json
import time

import cv2
from loguru import logger

from src import gamepad, input_exec
from src.capture import grab
from src.combat import Rejection, fallback_index, validate
from src.config import GAMEPAD, PATHS
from src.llm import ModelUnavailableError, ask_vlm
from src.memory import Memory, default_memory
from src.nav import plan
from src.perception import (
    HandScan,
    default_carddb,
    default_glyphbook,
    perceive,
    read_hud,
    scan_combat_hand,
    seek_card,
)
from src.schemas import ChoiceAction, CombatAction
from src.stall import Nudge, StallDetector
from src.states import GameState, NotTheGameError
from src.vision.minimap import Turn, read_minimap

# Mão conhecida, reaproveitada entre jogadas do mesmo turno. Refazer a travessia
# a cada carta jogada custava ~2.5s numa mão de 6 e não acrescentava informação:
# jogar uma carta só a remove da mão. Fica em nível de módulo porque os handlers
# recebem apenas `memory`; `forget_hand` zera em toda transição de estado.
_HAND: HandScan | None = None

_MAX_PARSE_FAILS = 3
_LOOP_SLEEP_S = 0.4
_MEMORY_RECENT_EVENTS = 8
_MAX_ILLEGAL_RETRIES = 2


def _memory_block(memory: Memory | None) -> str:
    """Bloco de contexto para injeção em prompts de decisão.

    Inclui resumo accordion + últimos eventos. Vazio se memory é None ou tem
    nada gravado — assim chamadores podem concatenar sem condicional.
    """
    if memory is None:
        return ""
    summary = memory.summary().strip()
    recent = memory.recent(_MEMORY_RECENT_EVENTS)
    if not summary and not recent:
        return ""
    parts = ["", "MEMÓRIA DA RUN (contexto acumulado):"]
    if summary and summary != "_(vazio)_":
        parts.append("Resumo:")
        parts.append(summary)
    if recent:
        parts.append("Eventos recentes:")
        parts.extend(f"- {e}" for e in recent)
    return "\n".join(parts)


def _combat_prompt(scan: HandScan, memory: Memory | None, complaint: str | None) -> str:
    hand = [
        {"indice": i, **c.model_dump()} for i, c in enumerate(scan.cards)
    ]
    parts = [
        (PATHS.prompts / "combat_decide.txt").read_text(encoding="utf-8"),
        _memory_block(memory),
        f"\n\nMANA DISPONÍVEL: {scan.mana if scan.mana is not None else 'desconhecida'}",
        _hp_line(scan),
        "\nMÃO (índices 0 = mais à esquerda):",
        json.dumps(hand, ensure_ascii=False, indent=2),
    ]
    if complaint:
        parts.append(
            f"\nATENÇÃO: sua resposta anterior foi rejeitada porque {complaint}. "
            "Escolha outra carta que respeite a mana e os índices acima."
        )
    return "\n".join(parts)


def _hp_line(scan: HandScan) -> str:
    """Linha de HP para o prompt, com aviso quando está baixo.

    Sem isso o modelo escolhe dano por padrão mesmo à beira da morte, e as cartas
    de armadura do deck nunca são jogadas.
    """
    if scan.hp is None:
        return ""
    current, maximum = scan.hp
    fraction = current / maximum if maximum else 1.0
    alert = "  ATENÇÃO: HP baixo, priorize armadura ou cura." if fraction <= 0.35 else ""
    return f"HP: {current}/{maximum}{alert}"


def _decide_combat(scan: HandScan, memory: Memory | None) -> CombatAction | None:
    """Pergunta a jogada ao modelo até vir uma legal. None se ele não conseguir.

    O modelo continua decidindo; o código só recusa o que o jogo recusaria. A
    contagem de rejeições é o sinal mais direto de quão bom o modelo é na tarefa.
    """
    complaint: str | None = None
    for attempt in range(_MAX_ILLEGAL_RETRIES + 1):
        result = ask_vlm(None, _combat_prompt(scan, memory, complaint), schema=CombatAction)
        action = CombatAction(**result)
        if action.acao == "finalizar_turno":
            return action
        verdict = validate(action.indice_alvo, scan.cards, scan.mana)
        if verdict is None:
            return action
        assert isinstance(verdict, Rejection)
        logger.warning(
            "Jogada ilegal (tentativa {}/{}): {}",
            attempt + 1, _MAX_ILLEGAL_RETRIES + 1, verdict.reason,
        )
        complaint = verdict.reason
    return None


def forget_hand() -> None:
    """Esquece a mão conhecida. Chamar em toda transição de estado."""
    global _HAND
    _HAND = None


def _current_hand() -> HandScan:
    """Mão conhecida com mana e HP atualizados, ou travessia se não houver.

    Reaproveitar é seguro porque `seek_card` confere a identidade da carta antes
    de confirmar: se a mão mudou de um jeito que não prevemos (carta comprada no
    meio do turno), ele falha e o passo seguinte refaz a travessia.
    """
    book = default_glyphbook()
    if _HAND is None:
        return scan_combat_hand(default_carddb(), book)
    mana, hp = read_hud(book)
    return _HAND.model_copy(update={"mana": mana, "hp": hp})


def handle_combat(memory: Memory | None = None) -> None:
    """Percorre a mão (ou reusa a conhecida), decide UMA jogada, valida e executa."""
    global _HAND
    scan = _current_hand()
    _HAND = scan
    if not scan.cards:
        logger.info("Sem cartas legíveis na mão — finalizando turno")
        input_exec.end_turn()
        forget_hand()
        return

    action = _decide_combat(scan, memory)
    if action is None:
        target = fallback_index(scan.cards, scan.mana)
        if target is None:
            logger.info("Nada jogável com {} de mana — finalizando turno", scan.mana)
            input_exec.end_turn()
            forget_hand()
            return
        logger.info("Modelo não achou jogada legal — regra de custo crescente: idx={}", target)
        motivo = "regra de reserva: custo crescente"
    elif action.acao == "finalizar_turno":
        logger.info("Finalizando turno: {}", action.motivo)
        input_exec.end_turn()
        _remember(memory, f"combate: finalizar turno ({action.motivo})", "combat")
        forget_hand()
        return
    else:
        target = action.indice_alvo
        motivo = action.motivo

    assert target is not None
    card = scan.cards[target]
    logger.info("Jogar '{}' idx={}, motivo: {}", card.nome, target, motivo)
    _play_card(target, scan, memory, motivo)


def _play_card(target: int, scan: HandScan, memory: Memory | None, motivo: str) -> None:
    """Posiciona o cursor pela IDENTIDADE da carta e só então confirma.

    Antes o agente calculava `alvo - cursor`, navegava e apertava X sem conferir
    onde o cursor tinha parado — com o índice errado, jogava a carta errada em
    silêncio. Agora ele anda até VER a carta escolhida em destaque.
    """
    global _HAND
    card = scan.cards[target]
    if seek_card(target, scan.cards, default_carddb()):
        _remember(memory, f"combate: jogou {card.nome} — {motivo}", "combat")
        input_exec.confirm()
        _HAND = scan.model_copy(
            update={"cards": scan.cards[:target] + scan.cards[target + 1 :]}
        )
        return
    logger.warning(
        "Não consegui posicionar o cursor em '{}' — refazendo a leitura da mão",
        card.nome,
    )
    _remember(memory, f"combate: cursor não chegou em {card.nome}", "combat")
    forget_hand()


def _remember(memory: Memory | None, event: str, state: str) -> None:
    if memory is not None:
        memory.append(event, state=state)


_TURN_ACTION = {
    Turn.FORWARD: input_exec.walk_forward,
    Turn.LEFT: input_exec.turn_left,
    Turn.RIGHT: input_exec.turn_right,
}


def handle_map(memory: Memory | None = None) -> None:
    """Lê o minimapa e dá UM passo rumo ao alvo.

    Sem VLM: "pra onde ir" está desenhado no minimapa, e uma busca em grafo
    responde exatamente. Cada passo recaptura, então erro não acumula.
    """
    frame = cv2.imread(str(grab(state="map")))
    minimap = read_minimap(frame)
    if minimap is None:
        logger.warning("Minimapa ilegível — andando pra frente pra destravar")
        input_exec.walk_forward()
        return

    step = plan(minimap)
    if step is None:
        logger.info("Nada alcançável no mapa — andando pra frente")
        input_exec.walk_forward()
        return

    logger.info(
        "Mapa: em {} olhando {} → {} (alvo: {})",
        minimap.player, minimap.facing.value, step.turn.value, step.reason,
    )
    _remember(memory, f"mapa: {step.turn.value} rumo a {step.reason}", "map")
    if step.turn is Turn.BACK:
        input_exec.turn_right()
        time.sleep(GAMEPAD.between_actions_s)
        input_exec.turn_right()
        return
    _TURN_ACTION[step.turn]()


def _decide_choice(
    state_json: str, contexto: str, memory: Memory | None = None
) -> ChoiceAction:
    prompt = (
        f"Você é o ESTRATEGISTA. Escolha a melhor opção do {contexto}.\n"
        "Considere sinergia com deck atual, custo, evitar redundância.\n"
        "Schema (responda APENAS JSON): "
        '{"indice_alvo": int, "motivo": str}\n'
        + _memory_block(memory)
        + "\n\nOPÇÕES:\n"
        + state_json
    )
    result = ask_vlm(None, prompt, schema=ChoiceAction)
    return ChoiceAction(**result)


def _choose_and_confirm(
    opcoes: list, cur: int | None, contexto: str, memory: Memory | None
) -> None:
    """Pede a escolha ao modelo, prende ao intervalo válido e navega até ela."""
    choice = _decide_choice(json.dumps(opcoes, ensure_ascii=False), contexto, memory)
    target = max(0, min(choice.indice_alvo, len(opcoes) - 1))
    if target != choice.indice_alvo:
        logger.warning("Índice {} fora de 0..{} — usando {}", choice.indice_alvo, len(opcoes) - 1, target)
    cursor = cur if cur is not None else len(opcoes) - 1
    logger.info("{}: escolhe idx={} ({})", contexto, target, choice.motivo)
    _remember(memory, f"{contexto}: idx={target} ({choice.motivo})", contexto)
    input_exec.select_and_confirm(target - cursor)


def handle_level_up(memory: Memory | None = None) -> None:
    perceived = perceive(str(grab(state="level_up")))
    if perceived.state is not GameState.LEVEL_UP:
        return
    data = perceived.data or {}
    opcoes = data.get("opcoes", [])
    if not opcoes:
        logger.warning("Level up sem opções legíveis — confirmando a selecionada")
        input_exec.confirm()
        return
    _choose_and_confirm(opcoes, data.get("indice_selecionada"), "level up", memory)


_CHEST_STATES = (GameState.CHEST, GameState.BOSS_CHEST, GameState.CHEST_CARD_TARGET)


def handle_chest(memory: Memory | None = None) -> None:
    perceived = perceive(str(grab(state="chest")))
    if perceived.state not in _CHEST_STATES:
        return
    data = perceived.data or {}
    tipo = data.get("tipo", "vazio")
    opcoes = data.get("opcoes", [])
    if tipo == "vazio" or not opcoes:
        input_exec.cancel()  # quadrado = sacar dinheiro
        return
    _choose_and_confirm(opcoes, data.get("indice_selecionada"), f"baú ({tipo})", memory)


def handle_stage_complete(memory: Memory | None = None) -> None:
    logger.info("Fase completa — andando pra frente")
    _remember(memory, "fase completa", "stage_complete")
    input_exec.walk_forward()


def handle_game_complete(memory: Memory | None = None) -> None:
    logger.info("Jogo concluído. Apertando X pra menu principal.")
    _remember(memory, "JOGO CONCLUÍDO", "game_complete")
    input_exec.confirm()


def handle_notice(memory: Memory | None = None) -> None:  # noqa: ARG001
    """Fecha um painel de aviso apertando X.

    Cobre telas que não são escolha nenhuma: "Nenhum controle detectado", e as
    duas confirmações que `jogo.md` descreve depois de escolher a evolução de
    carta. Sem este estado o prompt de diálogo era forçado a chutar uma das cinco
    telas de recompensa, e o agente tentava escolher onde não havia opção.
    """
    logger.info("Painel de aviso — confirmando")
    input_exec.confirm()


def handle_deck(memory: Memory | None = None) -> None:  # noqa: ARG001
    """Fecha a tela "Baralho".

    Não é um estado do jogo em si — é o jogador (ou um botão a mais) abrindo a
    visão do deck. Não há nada a decidir ali, só sair. Se o cancel não fechar, o
    antitravamento escalona.
    """
    logger.info("Tela Baralho — fechando")
    input_exec.cancel()


def handle_title(memory: Memory | None = None) -> None:  # noqa: ARG001
    logger.info("Tela de título — apertando X")
    input_exec.confirm()


def handle_menu(memory: Memory | None = None) -> None:  # noqa: ARG001
    logger.info("Menu — apertando cancel pra sair")
    input_exec.cancel()


def handle_game_over(memory: Memory | None = None) -> None:
    logger.warning("Game over — encerrando run")
    _remember(memory, "game over", "game_over")
    raise SystemExit(0)


_HANDLERS = {
    GameState.COMBAT: handle_combat,
    GameState.MAP: handle_map,
    GameState.LEVEL_UP: handle_level_up,
    GameState.CHEST: handle_chest,
    GameState.BOSS_CHEST: handle_chest,
    GameState.CHEST_CARD_TARGET: handle_chest,
    GameState.STAGE_COMPLETE: handle_stage_complete,
    GameState.GAME_COMPLETE: handle_game_complete,
    GameState.TITLE: handle_title,
    GameState.MENU: handle_menu,
    GameState.GAME_OVER: handle_game_over,
    GameState.DECK: handle_deck,
    GameState.NOTICE: handle_notice,
    GameState.SHOP: handle_menu,  # placeholder
}


_NUDGE_ACTION = {
    Nudge.CONFIRM: input_exec.confirm,
    Nudge.CANCEL: input_exec.cancel,
    Nudge.FORWARD: input_exec.walk_forward,
}


def _try_unstick(detector: StallDetector, memory: Memory) -> bool:
    """Empurra um botão quando a tela não muda. False quando esgotou as tentativas.

    Cobre telas que nenhum handler conhece — as duas confirmações que a evolução
    de carta abre, por exemplo. Sem isso o loop fica preso repetindo a mesma ação
    pra sempre.
    """
    nudge = detector.next_nudge()
    if nudge is None:
        return not detector.exhausted
    logger.warning("Tela não muda há {} passos — tentando {}", detector.patience, nudge.value)
    memory.append(f"destravando com {nudge.value}", state="stall")
    _NUDGE_ACTION[nudge]()
    return True


def _step(
    memory: Memory, last_state: GameState | None, detector: StallDetector
) -> GameState | None:
    frame_path = str(grab(state="loop"))
    detector.observe(cv2.imread(frame_path))
    if detector.stuck:
        if not _try_unstick(detector, memory):
            raise RuntimeError("tela travada e nenhum botão destravou")
        return last_state

    perceived = perceive(frame_path)
    if perceived.state is not last_state:
        memory.append(f"transição → {perceived.state.value}", state=perceived.state.value)
        forget_hand()  # a mão só vale dentro do mesmo combate
    handler = _HANDLERS.get(perceived.state)
    if handler is None:
        logger.warning("Sem handler para {}", perceived.state)
    else:
        handler(memory)
    return perceived.state


def loop(max_iters: int | None = None) -> int:
    PATHS.ensure()
    logger.info("Agente iniciado. Foque a janela do jogo.")
    time.sleep(GAMEPAD.boot_delay_s)

    memory = default_memory(summarize_fn=_summarize_via_vlm)
    memory.append("agente iniciado", state="boot")

    parse_fails = 0
    iters = 0
    last_state: GameState | None = None
    detector = StallDetector()
    try:
        while max_iters is None or iters < max_iters:
            iters += 1
            try:
                last_state = _step(memory, last_state, detector)
                parse_fails = 0
            except NotTheGameError as e:
                logger.error("{}", e)
                return 2
            except ModelUnavailableError as e:
                # Insistir não adianta: o servidor responde e o modelo não roda.
                logger.error("{}", e)
                return 3
            except (ValueError, RuntimeError) as e:
                parse_fails += 1
                logger.error("Falha no turno ({}/{}): {}", parse_fails, _MAX_PARSE_FAILS, e)
                memory.append(f"falha de percepção: {e}", state="error")
                if parse_fails >= _MAX_PARSE_FAILS:
                    logger.error("3 falhas seguidas — abortando")
                    return 1
            time.sleep(_LOOP_SLEEP_S)
        logger.info("Limite de iterações atingido")
        return 0
    finally:
        gamepad.reset()


def _summarize_via_vlm(body: str) -> str:
    """Sumariza eventos antigos da memória via VLM. Texto puro, sem schema."""
    prompt = (
        "Você é o cronista de uma run de Vampire Crawlers. Resuma os eventos "
        "abaixo em até 8 bullets curtos, mantendo decisões importantes, "
        "perdas de HP, cartas adquiridas e obstáculos. Em PT-BR.\n\n" + body
    )
    result = ask_vlm(None, prompt)
    return result if isinstance(result, str) else str(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="Loop principal do agente.")
    parser.add_argument("--iters", type=int, default=None, help="Máximo de iterações")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirma execução real (consome GPU pesado)",
    )
    args = parser.parse_args()

    if not args.confirm:
        logger.error("Use --confirm pra rodar (consome GPU pesado).")
        return 2

    return loop(max_iters=args.iters)


if __name__ == "__main__":
    raise SystemExit(main())
