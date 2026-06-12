"""Loop principal do agente. State machine com handlers por estado.

Filosofia: capture → detect_state → handler. Cada handler executa UMA micro-ação
via gamepad e retorna ao loop, que recaptura. Erros não acumulam — cada passo
se auto-corrige no próximo screenshot.
"""

from __future__ import annotations

import argparse
import json
import time

from loguru import logger

from src import gamepad, input_exec
from src.capture import grab
from src.config import GAMEPAD, PATHS
from src.llm import ask_vlm
from src.memory import Memory, default_memory
from src.perception import perceive, scan_combat_hand
from src.schemas import (
    ChoiceAction,
    CombatAction,
    MapDirection,
)
from src.states import GameState

_MAX_PARSE_FAILS = 3
_LOOP_SLEEP_S = 0.4
_MEMORY_RECENT_EVENTS = 8


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


def _decide_combat(
    state_json: str, scan_json: str, memory: Memory | None = None
) -> CombatAction:
    prompt_template = (PATHS.prompts / "combat_decide.txt").read_text(encoding="utf-8")
    prompt = (
        prompt_template
        + _memory_block(memory)
        + "\n\nESTADO ATUAL DO COMBATE:\n"
        + state_json
        + "\n\nSCAN DAS CARTAS NA MÃO (esquerda→direita):\n"
        + scan_json
    )
    result = ask_vlm(None, prompt, schema=CombatAction)
    return CombatAction(**result)


def _execute_combat_action(action: CombatAction, current_idx: int | None, total: int) -> None:
    """Move o cursor por travessia do índice atual até o alvo, e confirma.

    current_idx e indice_alvo são posições 0-based na mão (0 = esquerda).
    """
    if action.acao == "finalizar_turno":
        logger.info("Finalizando turno: {}", action.motivo)
        input_exec.end_turn()
        return

    target = action.indice_alvo
    if target is None or not 0 <= target < total:
        logger.warning("Ação inválida: indice_alvo={} total={}", target, total)
        return

    cur = current_idx if current_idx is not None else total - 1
    delta = target - cur
    logger.info(
        "Jogar carta idx={} (cursor={}), motivo: {}", target, cur, action.motivo
    )
    input_exec.select_and_confirm(delta)


def handle_combat(memory: Memory | None = None) -> None:
    """1) Captura → 2) percebe → 3) scan da mão → 4) decide → 5) executa."""
    frame = str(grab(state="combat_initial"))
    perceived = perceive(frame)
    if perceived.state is not GameState.COMBAT:
        logger.warning("Estado mudou de combat para {} no meio do handler", perceived.state)
        return

    data = perceived.data or {}
    total = int(data.get("total_cartas_visiveis", 0) or 0)
    if total <= 0:
        logger.info("Sem cartas visíveis — finalizando turno")
        input_exec.end_turn()
        return

    current_idx = data.get("indice_selecionada")
    if current_idx is None:
        # Cursor não está em nenhuma carta — força ← pra entrar no leque e recaptura.
        logger.info("Cursor não detectado — apertando ← pra entrar no leque")
        gamepad.tap_left(1)
        time.sleep(GAMEPAD.post_dpad_settle_s)
        frame = str(grab(state="combat_after_settle"))
        perceived = perceive(frame)
        data = perceived.data or {}
        total = int(data.get("total_cartas_visiveis", 0) or 0)
        current_idx = data.get("indice_selecionada")
        if current_idx is None:
            current_idx = total - 1
            logger.warning("Ainda sem cursor identificado — assumindo idx={}", current_idx)

    scan = scan_combat_hand(frame, total)
    # Após scan o cursor está na carta MAIS à esquerda (idx 0).
    cursor_after_scan = 0

    action = _decide_combat(
        state_json=json.dumps(data, ensure_ascii=False),
        scan_json=json.dumps([c.model_dump() for c in scan], ensure_ascii=False),
        memory=memory,
    )
    if memory is not None:
        memory.append(
            f"combate: {action.acao} idx={action.indice_alvo} ({action.motivo})",
            state="combat",
        )
    _execute_combat_action(action, cursor_after_scan, total)


def handle_map(memory: Memory | None = None) -> None:  # noqa: ARG001
    """Pergunta a direção do alvo e executa UMA micro-ação."""
    frame = str(grab(state="map"))
    perceived = perceive(frame)
    if perceived.state is not GameState.MAP:
        return
    direction = MapDirection(**(perceived.data or {}))
    logger.info("Direção do alvo: {} ({})", direction.direcao_alvo, direction.motivo)

    if direction.direcao_alvo == "frente":
        input_exec.walk_forward()
    elif direction.direcao_alvo == "esquerda":
        input_exec.turn_left()
    elif direction.direcao_alvo == "direita":
        input_exec.turn_right()
    elif direction.direcao_alvo == "atras":
        input_exec.turn_right()
        input_exec.turn_right()
    elif direction.direcao_alvo == "no_alvo":
        input_exec.walk_forward()


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


def handle_level_up(memory: Memory | None = None) -> None:
    frame = str(grab(state="level_up"))
    perceived = perceive(frame)
    if perceived.state is not GameState.LEVEL_UP:
        return
    data = perceived.data or {}
    opcoes = data.get("opcoes", [])
    cur = data.get("indice_selecionada")
    if cur is None:
        cur = len(opcoes) - 1

    choice = _decide_choice(json.dumps(opcoes, ensure_ascii=False), "level up", memory)
    delta = choice.indice_alvo - cur
    logger.info("Level up: escolhe idx={} ({})", choice.indice_alvo, choice.motivo)
    if memory is not None:
        memory.append(
            f"level up: idx={choice.indice_alvo} ({choice.motivo})", state="level_up"
        )
    input_exec.select_and_confirm(delta)


def handle_chest(memory: Memory | None = None) -> None:
    frame = str(grab(state="chest"))
    perceived = perceive(frame)
    if perceived.state not in (GameState.CHEST, GameState.BOSS_CHEST, GameState.CHEST_CARD_TARGET):
        return
    data = perceived.data or {}
    tipo = data.get("tipo", "vazio")
    if tipo == "vazio":
        input_exec.cancel()  # quadrado = sacar dinheiro
        return

    opcoes = data.get("opcoes", [])
    if not opcoes:
        input_exec.cancel()
        return

    choice = _decide_choice(
        json.dumps(opcoes, ensure_ascii=False), f"baú ({tipo})", memory
    )
    # Heurística: cursor inicia mais à direita por padrão. Quando a percepção
    # devolve indice_selecionada (chest_card_target), respeita ela.
    cur = data.get("indice_selecionada")
    if cur is None:
        cur = len(opcoes) - 1
    delta = choice.indice_alvo - cur
    logger.info("Baú {}: escolhe idx={} ({})", tipo, choice.indice_alvo, choice.motivo)
    if memory is not None:
        memory.append(
            f"baú {tipo}: idx={choice.indice_alvo} ({choice.motivo})", state="chest"
        )
    input_exec.select_and_confirm(delta)


def handle_stage_complete(memory: Memory | None = None) -> None:  # noqa: ARG001
    logger.info("Fase completa — andando pra frente")
    input_exec.walk_forward()


def handle_game_complete(memory: Memory | None = None) -> None:  # noqa: ARG001
    logger.info("Jogo concluído. Apertando X pra menu principal.")
    input_exec.confirm()


def handle_title(memory: Memory | None = None) -> None:  # noqa: ARG001
    logger.info("Tela de título — apertando X")
    input_exec.confirm()


def handle_menu(memory: Memory | None = None) -> None:  # noqa: ARG001
    logger.info("Menu — apertando cancel pra sair")
    input_exec.cancel()


def handle_game_over(memory: Memory | None = None) -> None:  # noqa: ARG001
    logger.warning("Game over — encerrando run")
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
    GameState.SHOP: handle_menu,  # placeholder
}


def loop(max_iters: int | None = None) -> int:
    PATHS.ensure()
    logger.info("Agente iniciado. Foque a janela do jogo.")
    time.sleep(GAMEPAD.boot_delay_s)

    memory = default_memory(summarize_fn=_summarize_via_vlm)
    memory.append("agente iniciado", state="boot")

    parse_fails = 0
    iters = 0
    last_state = None
    try:
        while True:
            if max_iters is not None and iters >= max_iters:
                logger.info("Limite de iterações atingido")
                return 0
            iters += 1

            try:
                frame = str(grab(state="loop"))
                perceived = perceive(frame)
                if perceived.state is not last_state:
                    memory.append(f"transição → {perceived.state.value}", state=perceived.state.value)
                    last_state = perceived.state
                handler = _HANDLERS.get(perceived.state)
                if handler is None:
                    logger.warning("Sem handler para {}", perceived.state)
                else:
                    handler(memory)
                parse_fails = 0
            except (ValueError, RuntimeError) as e:
                parse_fails += 1
                logger.error("Falha no turno ({}/{}): {}", parse_fails, _MAX_PARSE_FAILS, e)
                memory.append(f"falha de percepção: {e}", state="error")
                if parse_fails >= _MAX_PARSE_FAILS:
                    logger.error("3 falhas seguidas — abortando")
                    return 1

            time.sleep(_LOOP_SLEEP_S)
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
