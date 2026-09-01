"""Sessão de captura rotulada: gera o conjunto de regressão do projeto.

Você joga normalmente. A cada tela interessante, aperta a tecla do estado
correspondente; a ferramenta captura o frame, grava o rótulo e só ENTÃO mostra o
que a CV teria respondido. Divergência aparece na hora, então dá pra caçar os
casos difíceis de propósito em vez de torcer pra aparecerem.

**Não é pra rotular o jogo inteiro.** O que faz uma suíte de regressão valer é
variedade, não volume: umas dez telas de combate diferentes valem mais que cem
frames do mesmo turno. A meta por estado está em `_META` e a sessão mostra o
quanto falta a cada captura — quando fecha, pode parar.

A saída (`dataset/`) é versionável e vira teste automatizado — é o que permite
dizer "contagem 98% em 60 frames" em vez de "parece melhor". É também o conjunto
que a comparação entre modelos vai usar.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
from loguru import logger

from src.capture import grab
from src.config import PROJECT_ROOT
from src.perception import default_glyphbook, read_hp_hybrid, read_mana_hybrid
from src.vision.cards import detect_card_slots, detect_choice_slots, read_costs
from src.vision.hud import read_hp, read_mana
from src.vision.icons import find_icons
from src.vision.minimap import read_minimap
from src.vision.screen import Verdict, signature
from src.window import find_game_window, game_is_visible

DATASET_DIR = PROJECT_ROOT / "dataset"
LABELS_FILE = DATASET_DIR / "labels.jsonl"

# Teclas escolhidas pela inicial do estado, sem colisão.
_KEYS: dict[str, str] = {
    "c": "combat",
    "m": "map",
    "l": "level_up",
    "b": "chest",
    "n": "boss_chest",
    "t": "chest_card_target",
    "s": "shop",
    "f": "stage_complete",
    "v": "game_complete",
    "i": "title",
    "e": "menu",
    "g": "game_over",
}

# Quantos frames por estado bastam. Combate e mapa levam mais porque dominam o
# loop e porque são exatamente os dois que o caminho antigo confundia entre si
# (ADR-077). Os estados raros levam poucos: um frame já pega a assinatura de cor.
_META: dict[str, int] = {
    "combat": 12,
    "map": 10,
    "level_up": 4,
    "shop": 3,
    "chest": 3,
    "boss_chest": 2,
    "chest_card_target": 2,
    "stage_complete": 1,
    "game_complete": 1,
    "title": 1,
    "menu": 1,
    "game_over": 1,
}


@dataclass(frozen=True)
class Pergunta:
    """Uma pergunta de detalhe e o campo da CV que ela serve de gabarito.

    O par `campo`/`cv` é a regra que impede a lista de inchar: só entra pergunta
    que tem do outro lado um número que a CV já produz. Pergunta sem `cv` vira
    opinião solta no dataset — custa o tempo de quem responde e não mede nada.
    """

    campo: str
    cv: str
    texto: str
    parse: Callable[[str], tuple[object, bool]]
    onde: str


def _num(raw: str) -> tuple[object, bool]:
    return (int(raw), True) if raw.isdigit() else (None, False)


def _selecionada(raw: str) -> tuple[object, bool]:
    """Pergunta em 1-based, grava em 0-based. 'n' = nenhuma selecionada.

    "Nenhuma" é uma RESPOSTA, não uma falta: o cursor sai da mão sempre que o
    jogador está sobre "Finalizar turno" ou "Jogar todas". Tratar as duas coisas
    como o mesmo `None` apagaria do gabarito o caso mais comum de
    `selected_idx is None` — justamente o que precisa ser medido.
    """
    if raw in ("n", "nenhuma"):
        return None, True
    if raw.isdigit() and int(raw) >= 1:
        return int(raw) - 1, True
    return None, False


def _par_hp(raw: str) -> tuple[object, bool]:
    """Lista, não tupla: o JSONL grava lista, e comparar tupla com lista dá
    ERROU em cima de dois valores iguais."""
    partes = raw.split("/")
    if len(partes) == 2 and all(p.strip().isdigit() for p in partes):
        return [int(p) for p in partes], True
    return None, False


def _sim_nao(raw: str) -> tuple[object, bool]:
    if raw in ("s", "sim"):
        return True, True
    if raw in ("n", "nao", "não"):
        return False, True
    return None, False


def _direcao(raw: str) -> tuple[object, bool]:
    rumo = {"n": "norte", "s": "sul", "l": "leste", "o": "oeste"}
    return (rumo[raw[0]], True) if raw and raw[0] in rumo else (None, False)


def _lista_de_custos(raw: str) -> tuple[object, bool]:
    """Uma pergunta só pra mão inteira: digitar seis números separados é mais
    rápido que responder seis perguntas, e dá o gabarito na ordem certa."""
    partes = [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip()]
    if partes and all(p.isdigit() for p in partes):
        return [int(p) for p in partes], True
    return None, False


_ESCOLHA = (
    Pergunta(
        "offered",
        "cv_choice_cards",
        "quantas cartas na oferta?",
        _num,
        "as cartas do painel central",
    ),
    Pergunta(
        "cursor",
        "cv_choice_cursor",
        "qual está selecionada? (1 = a da esquerda, n = nenhuma)",
        _selecionada,
        "aqui a selecionada é a mais ALTA, não a maior",
    ),
)

# Estados de escolha: mesmo painel, mesmas duas perguntas.
_DIALOGOS = ("level_up", "chest", "boss_chest", "chest_card_target", "shop")

_PERGUNTAS: dict[str, tuple[Pergunta, ...]] = {
    "combat": (
        Pergunta(
            "hand_size",
            "cv_hand_size",
            "cartas na mão?",
            _num,
            "conte TODAS as do leque, inclusive as tapadas pela vizinha",
        ),
        Pergunta(
            "cursor",
            "cv_cursor",
            "carta levantada? (1 = a da esquerda, n = nenhuma)",
            _selecionada,
            "a selecionada sobe acima das outras e o círculo de custo dela fica "
            "maior, pulsando entre azul e magenta",
        ),
        Pergunta(
            "costs",
            "cv_costs",
            "custo de cada carta, da esquerda pra direita? (ex.: 1,1,1,2,0,0)",
            _lista_de_custos,
            "o algarismo dentro do círculo de cada carta — o que ela CUSTA",
        ),
        Pergunta(
            "mana",
            "cv_mana",
            "mana disponível?",
            _num,
            "o número no orbe azul à direita do leque — o que você TEM",
        ),
        Pergunta(
            "hp",
            "cv_hp",
            "HP? (atual/máximo, ex.: 61/61)",
            _par_hp,
            "os dois números dentro do coração, à esquerda do leque",
        ),
    ),
    "map": (
        Pergunta(
            "facing",
            "cv_facing",
            "pra onde a seta aponta? (n/s/l/o)",
            _direcao,
            "a seta azul do minimapa",
        ),
        Pergunta(
            "enemies",
            "cv_enemies",
            "quantas caveiras no minimapa?",
            _num,
            "só caveiras; baú, interrogação e o ícone de chefe não contam",
        ),
        Pergunta(
            "boss",
            "cv_boss",
            "o ícone de chefe aparece? (s/n)",
            _sim_nao,
            "só se estiver visível no minimapa agora",
        ),
    ),
    **dict.fromkeys(_DIALOGOS, _ESCOLHA),
}


def _como_responder() -> str:
    """O texto de ajuda sai da mesma tabela que faz as perguntas.

    Escrito à mão, ele descreveria a versão anterior das perguntas na primeira
    vez que alguém mudasse a tabela — e ajuda errada é pior que ajuda nenhuma.
    """
    linhas = ["", "O QUE VOU PERGUNTAR, E ONDE OLHAR", ""]
    for estado, perguntas in _PERGUNTAS.items():
        if estado in _DIALOGOS and estado != _DIALOGOS[0]:
            continue
        titulo = "telas de escolha (level up, baú, loja)" if estado in _DIALOGOS else estado
        linhas.append(f"  {titulo}")
        linhas += [f"    {p.campo:10} {p.onde}" for p in perguntas]
        linhas.append("")
    linhas += [
        '  Enter em branco = "não sei": a pergunta fica sem gabarito, e nem conta',
        "  como acerto nem como erro. Responder errado é pior que pular.",
        "",
        "O palpite da CV só aparece DEPOIS que você responde, de propósito: se",
        "aparecesse antes você concordaria com ele, e o gabarito passaria a medir",
        "a CV contra ela mesma.",
        "",
    ]
    return "\n".join(linhas)


def _menu() -> str:
    linhas = [f"  [{k}] {v}" for k, v in _KEYS.items()]
    return "\n".join(linhas) + "\n  [q] sair"


def _read_key() -> str:
    """Uma tecla, sem Enter. Cai pra input() fora do Windows."""
    try:
        import msvcrt
    except ImportError:
        return (input("estado> ").strip() or "?")[0]
    return msvcrt.getch().decode("utf-8", errors="ignore").lower()


def _observed(frame_path: Path) -> dict:
    """O que a CV enxerga — gravado junto pra medir divergência depois.

    Mana e HP saem do caminho **local** (`read_mana`/`read_hp`), não do híbrido:
    o híbrido cai pro modelo quando o glifo é novo, e 800ms por frame travaria a
    sessão. Pior, ensinaria o livro de glifos a partir de uma captura sem foco —
    exatamente o que a ADR-074 proíbe. Aqui, algarismo desconhecido vira `None`,
    que a comparação trata como "a CV não sabe".
    """
    frame = cv2.imread(str(frame_path))
    sig = signature(frame)
    slots = detect_card_slots(frame)
    book = default_glyphbook()
    hp = read_hp(frame, book)
    return {
        "cv_verdict": sig.verdict.value,
        "cv_parchment": sig.parchment,
        "cv_slate": sig.slate,
        "cv_cards": slots.visible_total,
        "cv_hand_size": slots.hand_size,
        "cv_hidden": slots.hidden_idx,
        "cv_cursor": slots.selected_idx,
        "cv_costs": read_costs(frame, slots, book),
        **_observed_choice(frame),
        "cv_mana": read_mana(frame, book),
        "cv_hp": list(hp) if hp else None,
        **_observed_minimap(read_minimap(frame)),
    }


def _observed_choice(frame) -> dict:
    """O painel central das telas de escolha tem detector próprio.

    `detect_card_slots` olha a caixa da MÃO, que só se sobrepõe em parte ao
    painel; usar a contagem da mão como gabarito de "quantas cartas na oferta"
    mediria a caixa errada e culparia a CV por isso.
    """
    escolha = detect_choice_slots(frame)
    return {"cv_choice_cards": escolha.visible_total, "cv_choice_cursor": escolha.selected_idx}


def _observed_minimap(minimap) -> dict:
    if minimap is None:
        return {"cv_facing": None, "cv_enemies": None, "cv_boss": None}
    icones = find_icons(minimap.gray, minimap.arrow_side)
    contagem = collections.Counter(ic.kind.value for ic in icones)
    return {
        "cv_facing": minimap.facing.value,
        "cv_enemies": contagem.get("inimigo", 0),
        "cv_boss": contagem.get("chefe", 0) > 0,
    }


def _append(record: dict) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with LABELS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _pode_capturar() -> bool:
    """Guarda da rotulagem: o jogo precisa estar VISÍVEL, não reconhecido.

    Quando o Win32 não localiza a janela, `game_is_visible` devolve None — aí
    não há resposta do sistema e sobra a checagem por conteúdo, que é fraca mas
    é melhor que nada. Nesse caminho o ponto cego volta, e o aviso diz isso.
    """
    visivel = game_is_visible()
    if visivel is True:
        return True
    if visivel is False:
        logger.error(
            "A janela do jogo está COBERTA nos pixels que seriam capturados. "
            "Ponha o jogo e o terminal lado a lado (ou em monitores diferentes) — "
            "o jogo não precisa estar focado, mas precisa aparecer inteiro."
        )
        return False
    logger.warning(
        "Janela do jogo não localizada; caindo na checagem por conteúdo, que "
        "recusa telas que a CV ainda não reconhece (menu, loja, título)."
    )
    amostra = grab(state="label_check")
    fora = signature(cv2.imread(str(amostra))).verdict is Verdict.NOT_GAME
    amostra.unlink(missing_ok=True)
    if fora:
        logger.error("A captura não parece ser o jogo. Ajeite as janelas e tente de novo.")
    return not fora


def capture_labeled(state: str) -> Path | None:
    """Captura o frame atual e move pro dataset. None quando não é o jogo.

    Captura ANTES de perguntar os detalhes: quem responde precisa estar falando
    do frame que foi realmente salvo, não do que lembra de ter visto antes de
    digitar. Perguntar primeiro também jogaria fora a resposta quando a captura
    é recusada.

    **Aqui o foco não serve de guarda**: pra ler a tecla, o terminal precisa
    estar focado, então o jogo nunca está. Quem responde é o SISTEMA, via
    `game_is_visible` — a janela do jogo está no topo nos pixels que vão ser
    capturados? Sem alguma checagem, um jogo atrás do terminal produziria um
    dataset inteiro de prints do terminal rotulados como "combate".

    Não dá mais pra perguntar "o frame PARECE o jogo?" à CV, como antes. Aquilo
    era circular: exigia que a CV reconhecesse a tela, e as telas que mais
    precisam ser rotuladas são justamente as que ela ainda não reconhece — menu,
    loja, título, game over. O guarda recusava exatamente o material necessário
    pra consertar o próprio ponto cego (ADR-084).
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if not _pode_capturar():
        return None
    shot = grab(state=f"label_{state}")
    target = DATASET_DIR / shot.name
    shot.replace(target)
    return target


def registrar(target: Path, state: str, detalhes: dict) -> dict:
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "file": target.name,
        "state": state,
        **detalhes,
        **_observed(target),
    }
    _append(record)
    return record


def _perguntar_detalhes(state: str) -> dict:
    detalhes: dict = {}
    for p in _PERGUNTAS.get(state, ()):
        valor, respondeu = p.parse(input(f"   {p.texto} ").strip().lower())
        detalhes[p.campo] = valor
        detalhes[f"{p.campo}_known"] = respondeu
    return detalhes


def _sabe(record: dict, campo: str) -> bool:
    """Rótulos antigos não tinham o par `_known`: valor presente já era resposta."""
    if f"{campo}_known" in record:
        return bool(record[f"{campo}_known"])
    return record.get(campo) is not None


def _confere(record: dict, p: Pergunta) -> str:
    if not _sabe(record, p.campo):
        return ""
    rotulo, obtido = record.get(p.campo), record.get(p.cv)
    marca = "ok" if rotulo == obtido else "ERROU"
    return f" {p.campo}={_mostrar(rotulo)}/cv={_mostrar(obtido)} {marca}"


def _mostrar(v: object) -> str:
    if v is None:
        return "nenhuma"
    if isinstance(v, list):
        return "/".join(str(x) for x in v)
    return str(v)


def _report(record: dict, feito: collections.Counter) -> None:
    estado = record["state"]
    agree = record["cv_verdict"] in (estado, "dialog")
    detalhes = "".join(_confere(record, p) for p in _PERGUNTAS.get(estado, ()))
    logger.info(
        "{} rotulado={} cv={}{}  [{}/{}]",
        "ok " if agree else "DIVERGE",
        estado,
        record["cv_verdict"],
        detalhes,
        feito[estado],
        _META.get(estado, 0),
    )


def _contagem_gravada() -> collections.Counter:
    if not LABELS_FILE.is_file():
        return collections.Counter()
    linhas = LABELS_FILE.read_text(encoding="utf-8").splitlines()
    return collections.Counter(json.loads(ln)["state"] for ln in linhas if ln.strip())


# O que precisa VARIAR entre os frames de um mesmo estado. Sem isso a meta
# parece "tire 12 fotos do combate", e 12 fotos do mesmo turno medem uma
# situação só — o detector passaria na suíte sem nunca ter sido testado.
_VARIEDADE: dict[str, str] = {
    "combat": "mãos de tamanhos diferentes, com e sem carta levantada, mana alta e baixa",
    "map": "corredor, sala, cruzamento; virado pros quatro lados; com e sem chefe à vista",
    "level_up": "ofertas diferentes, cursor em posições diferentes",
    "shop": "com e sem dinheiro pra comprar",
    "chest": "cursor em posições diferentes",
    "boss_chest": "cursor em posições diferentes",
    # "Inserir joia em uma carta": mostra o DECK, não a mão, e aparece tanto
    # depois do baú quanto depois de um level up com carta de bônus — o nome
    # `chest_` engana. Marque com [t] nos dois casos.
    "chest_card_target": "decks de tamanhos diferentes, vindo do baú E do level up",
}


def _tabela_cobertura(feito: collections.Counter) -> str:
    faltando = [(e, n) for e, n in _META.items() if feito.get(e, 0) < n]
    if not faltando:
        return "Cobertura completa. Pode parar quando quiser."
    linhas = [
        "Ainda falta. NÃO é pra rotular o jogo todo, e nem tirar N fotos da mesma",
        "tela: o que mede alguma coisa é a VARIAÇÃO entre os frames.",
        "",
    ]
    for estado, n in faltando:
        variar = _VARIEDADE.get(estado, "")
        sufixo = f"   variar: {variar}" if variar else ""
        linhas.append(f"  {estado:20} {feito.get(estado, 0):>2}/{n}{sufixo}")
    return "\n".join(linhas)


def session(ask_details: bool) -> int:
    print("Sessão de rotulagem.")
    print(
        "O jogo precisa ficar VISÍVEL enquanto você digita aqui — lado a lado ou\n"
        "em outro monitor. Focado ele não vai estar (o terminal precisa do foco\n"
        "pra ler a tecla), e não precisa.\n"
    )
    if not _pode_capturar():
        logger.error("Ajeite as janelas e rode de novo — sem isso o dataset sairia errado.")
        return 2
    if ask_details:
        print(_como_responder())
    feito = _contagem_gravada()
    print(_tabela_cobertura(feito) + "\n")
    print(_menu())
    return _laco(ask_details, feito)


def _laco(ask_details: bool, feito: collections.Counter) -> int:
    total = 0
    while True:
        key = _read_key()
        if key == "q":
            break
        state = _KEYS.get(key)
        if state is None:
            continue
        target = capture_labeled(state)
        if target is None:
            continue
        detalhes = _perguntar_detalhes(state) if ask_details else {}
        feito[state] += 1
        _report(registrar(target, state, detalhes), feito)
        total += 1
    print(f"\n{total} frames gravados nesta sessão, em {DATASET_DIR}")
    print(_tabela_cobertura(feito))
    return 0


def watch(interval_s: float, samples: int) -> int:
    """Observa o jogo sem tocar nele: captura periódica e relatório do que a CV vê.

    Existe pra validar o sensor contra o jogo rodando sem o risco de o agente
    assumir o controle. Não emite input nenhum — nem cria o gamepad virtual.

    Usa o caminho híbrido de leitura de mana, então também aquece o livro de
    glifos: depois de ver cada algarismo duas vezes, a leitura passa a ser local.

    **Pula a amostra quando o jogo não está em primeiro plano.** Observar não
    precisa de foco por segurança, mas precisa por validade: `mss` captura uma
    região da tela, então sem foco cada leitura é do que estiver por cima. Uma
    sessão assim reportou mana de 104 e 12 lendo números de outra janela — e
    pior, ENSINOU esses algarismos ao livro de glifos. O portão de dois votos
    conteve o estrago, mas dado ruim não deve entrar de propósito.
    """
    print(f"Observando a cada {interval_s}s por {samples} amostras. Jogue normalmente.\n")
    print(f"{'#':>3} {'estado':10} {'cartas':>6} {'cursor':>6} {'mana':>5} {'HP':>7}  minimapa")
    fora_de_foco = 0
    for i in range(1, samples + 1):
        if not find_game_window().foreground:
            fora_de_foco += 1
            print(f"{i:>3} {'(sem foco)':10} — traga a janela do jogo pra frente")
            if i < samples:
                time.sleep(interval_s)
            continue
        shot = grab(state="watch")
        frame = cv2.imread(str(shot))
        sig = signature(frame)
        slots = detect_card_slots(frame)
        book = default_glyphbook()
        mana = read_mana_hybrid(frame, book)
        hp = read_hp_hybrid(frame, book)
        minimap = read_minimap(frame)
        mm = "-"
        if minimap is not None:
            icons = find_icons(minimap.gray, minimap.arrow_side)
            counts = collections.Counter(ic.kind.value for ic in icons)
            mm = f"{minimap.facing.value}, {counts.get('inimigo', 0)} inimigos, chefe={counts.get('chefe', 0)}"
        print(
            f"{i:>3} {sig.verdict.value:10} {slots.visible_total:>6} "
            f"{str(slots.selected_idx):>6} {str(mana):>5} {str(hp):>7}  {mm}"
        )
        if i < samples:
            time.sleep(interval_s)
    if fora_de_foco:
        print(
            f"\n{fora_de_foco} de {samples} amostras puladas por falta de foco. "
            "Sem o jogo em primeiro plano, a captura pega o que estiver por cima."
        )
    return 0


def _acuracia(records: list[dict], p: Pergunta) -> str:
    """Concordância só entre os frames que têm gabarito daquele campo.

    Frame sem resposta não é acerto nem erro: contar como erro puniria a CV por
    uma pergunta que ninguém respondeu, e contar como acerto inflaria o número.
    """
    com_rotulo = [r for r in records if _sabe(r, p.campo)]
    if not com_rotulo:
        return f"{'sem gabarito':>16}"
    ok = sum(1 for r in com_rotulo if r.get(p.campo) == r.get(p.cv))
    return f"{ok:>4}/{len(com_rotulo):<4} {100 * ok / len(com_rotulo):>4.0f}%"


def _tabela_detalhes(by_state: dict[str, list[dict]]) -> str:
    """Onde a CV erra, campo por campo. É pra isso que a rotulagem existe."""
    linhas: list[str] = []
    for estado, perguntas in _PERGUNTAS.items():
        rows = by_state.get(estado)
        if not rows:
            continue
        linhas.append(f"\n{estado}")
        linhas += [f"  {p.campo:12} {_acuracia(rows, p)}" for p in perguntas]
    return "\n".join(linhas)


def summary() -> int:
    if not LABELS_FILE.is_file():
        print("Nenhum rótulo ainda. Rode `python -m src.label` pra criar.")
        return 1
    records = [
        json.loads(ln) for ln in LABELS_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    by_state: dict[str, list[dict]] = {}
    for r in records:
        by_state.setdefault(r["state"], []).append(r)

    print(f"{len(records)} frames rotulados\n")
    print(f"{'estado':20} {'n':>4} {'meta':>5} {'CV concorda':>12}")
    for state, rows in sorted(by_state.items()):
        agree = sum(1 for r in rows if r["cv_verdict"] in (state, "dialog"))
        meta = _META.get(state, 0)
        print(f"{state:20} {len(rows):>4} {meta:>5} {100 * agree / len(rows):>11.0f}%")

    detalhes = _tabela_detalhes(by_state)
    if detalhes.strip():
        print("\nCV x gabarito, só nos frames com resposta:" + detalhes)
    print("\n" + _tabela_cobertura(collections.Counter({k: len(v) for k, v in by_state.items()})))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Captura rotulada pro dataset de regressão.")
    parser.add_argument("--summary", action="store_true", help="Resume o dataset já gravado e sai.")
    parser.add_argument(
        "--watch",
        type=int,
        metavar="N",
        help="Observa N amostras sem emitir input nenhum e relata o que a CV vê.",
    )
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument(
        "--details",
        action="store_true",
        help="Em combate, também pergunta tamanho da mão e carta levantada.",
    )
    args = parser.parse_args()

    if args.summary:
        return summary()
    if args.watch:
        return watch(args.interval, args.watch)
    if not sys.stdin.isatty():
        logger.error("Rode num terminal interativo — a rotulagem lê teclas.")
        return 2
    return session(ask_details=args.details)


if __name__ == "__main__":
    raise SystemExit(main())
