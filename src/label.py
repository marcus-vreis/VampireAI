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
from datetime import UTC, datetime
from pathlib import Path

import cv2
from loguru import logger

from src.capture import grab
from src.config import PROJECT_ROOT
from src.perception import default_glyphbook, read_hp_hybrid, read_mana_hybrid
from src.vision.cards import detect_card_slots
from src.vision.icons import find_icons
from src.vision.minimap import read_minimap
from src.vision.screen import Verdict, signature
from src.window import find_game_window

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

_COMO_RESPONDER = """
COMO RESPONDER OS DETALHES (só aparecem em combate)

  cartas na mão   conte TODAS as cartas do leque, inclusive as que estão
                  parcialmente tapadas pela vizinha da esquerda.

  carta levantada a selecionada sobe acima das outras e o círculo de custo dela
                  fica maior, pulsando entre azul e magenta. Responda contando
                  da esquerda: 1 = a primeira. Se NENHUMA estiver levantada
                  (acontece sempre que o cursor está em "Finalizar turno" ou
                  "Jogar todas"), responda 'n' -- isso é uma resposta correta,
                  não uma falta.

  Enter em branco significa "não sei", e a pergunta fica sem gabarito.

O palpite da CV só aparece DEPOIS que você responde, de propósito: se aparecesse
antes você concordaria com ele, e o gabarito passaria a medir a CV contra ela
mesma.
"""


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
    """O que a CV enxerga — gravado junto pra medir divergência depois."""
    frame = cv2.imread(str(frame_path))
    sig = signature(frame)
    slots = detect_card_slots(frame)
    minimap = read_minimap(frame)
    return {
        "cv_verdict": sig.verdict.value,
        "cv_parchment": sig.parchment,
        "cv_slate": sig.slate,
        "cv_cards": slots.visible_total,
        "cv_cursor": slots.selected_idx,
        "cv_facing": minimap.facing.value if minimap else None,
    }


def _append(record: dict) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with LABELS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def capture_labeled(state: str) -> Path | None:
    """Captura o frame atual e move pro dataset. None quando não é o jogo.

    Captura ANTES de perguntar os detalhes: quem responde precisa estar falando
    do frame que foi realmente salvo, não do que lembra de ter visto antes de
    digitar. Perguntar primeiro também jogaria fora a resposta quando a captura
    é recusada.

    **Aqui o foco não serve de guarda**: pra ler a tecla, o terminal precisa
    estar focado, então o jogo nunca está. O que vale é perguntar se o frame
    PARECE o jogo — e a assinatura de CV já responde isso.

    Sem esta checagem, um jogo atrás do terminal produziria um dataset inteiro de
    prints do terminal rotulados como "combate", e o gabarito que deveria medir o
    sensor mediria outra coisa.
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    shot = grab(state=f"label_{state}")
    if signature(cv2.imread(str(shot))).verdict is Verdict.NOT_GAME:
        shot.unlink(missing_ok=True)
        logger.error(
            "A captura não é o jogo. Deixe a janela do Vampire Crawlers VISÍVEL "
            "(lado a lado ou em outro monitor) — ela não precisa estar focada, "
            "mas precisa aparecer na tela."
        )
        return None
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


def _perguntar_quantidade(prompt: str) -> tuple[int | None, bool]:
    raw = input(prompt).strip()
    return (int(raw), True) if raw.isdigit() else (None, False)


def _perguntar_selecionada(prompt: str) -> tuple[int | None, bool]:
    """Pergunta em 1-based, grava em 0-based. 'n' = nenhuma levantada.

    "Nenhuma" é uma RESPOSTA, não uma falta: o cursor sai da mão sempre que o
    jogador está sobre "Finalizar turno" ou "Jogar todas". Tratar as duas coisas
    como o mesmo `None` apagaria do gabarito o caso mais comum de
    `selected_idx is None` — justamente o que precisa ser medido.
    """
    raw = input(prompt).strip().lower()
    if raw in ("n", "nenhuma"):
        return None, True
    if raw.isdigit() and int(raw) >= 1:
        return int(raw) - 1, True
    return None, False


def _perguntar_detalhes() -> dict:
    mao, mao_ok = _perguntar_quantidade("   cartas na mão? (número, Enter = não sei) ")
    cur, cur_ok = _perguntar_selecionada(
        "   carta levantada? (1 = primeira da esquerda, n = nenhuma, Enter = não sei) "
    )
    return {"hand_size": mao, "hand_size_known": mao_ok, "cursor": cur, "cursor_known": cur_ok}


def _sabe(record: dict, campo: str) -> bool:
    """Rótulos antigos não tinham o par `_known`: valor presente já era resposta."""
    if f"{campo}_known" in record:
        return bool(record[f"{campo}_known"])
    return record.get(campo) is not None


def _confere(record: dict, campo: str, cv: str) -> str:
    if not _sabe(record, campo):
        return ""
    rotulo, obtido = record.get(campo), record.get(cv)
    marca = "ok" if rotulo == obtido else "ERROU"
    return f" {campo}={_mostrar(rotulo)} cv={_mostrar(obtido)} {marca}"


def _mostrar(v: int | None) -> str:
    return "nenhuma" if v is None else str(v)


def _report(record: dict, feito: collections.Counter) -> None:
    agree = record["cv_verdict"] in (record["state"], "dialog")
    mark = "ok " if agree else "DIVERGE"
    estado = record["state"]
    meta = _META.get(estado, 0)
    logger.info(
        "{} rotulado={} cv={}{}{}  [{}/{}]",
        mark,
        estado,
        record["cv_verdict"],
        _confere(record, "hand_size", "cv_cards"),
        _confere(record, "cursor", "cv_cursor"),
        feito[estado],
        meta,
    )


def _contagem_gravada() -> collections.Counter:
    if not LABELS_FILE.is_file():
        return collections.Counter()
    linhas = LABELS_FILE.read_text(encoding="utf-8").splitlines()
    return collections.Counter(json.loads(ln)["state"] for ln in linhas if ln.strip())


def _tabela_cobertura(feito: collections.Counter) -> str:
    faltando = [(e, n) for e, n in _META.items() if feito.get(e, 0) < n]
    if not faltando:
        return "Cobertura completa. Pode parar quando quiser."
    linhas = ["Ainda falta (você NÃO precisa rotular o jogo todo):"]
    linhas += [f"  {e:20} {feito.get(e, 0):>2}/{n}" for e, n in faltando]
    return "\n".join(linhas)


def session(ask_details: bool) -> int:
    print("Sessão de rotulagem.")
    print(
        "O jogo precisa ficar VISÍVEL enquanto você digita aqui — lado a lado ou\n"
        "em outro monitor. Focado ele não vai estar (o terminal precisa do foco\n"
        "pra ler a tecla), e não precisa.\n"
    )
    if signature(cv2.imread(str(grab(state="label_check")))).verdict is Verdict.NOT_GAME:
        logger.error(
            "A captura não está mostrando o jogo. Ajeite as janelas e rode de novo — "
            "sem isso o dataset inteiro sairia com prints do terminal."
        )
        return 2
    if ask_details:
        print(_COMO_RESPONDER)
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
        detalhes = _perguntar_detalhes() if ask_details and state == "combat" else {}
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


def _acuracia(records: list[dict], campo: str, cv: str) -> str:
    """Concordância só entre os frames que têm gabarito daquele campo."""
    com_rotulo = [r for r in records if _sabe(r, campo)]
    if not com_rotulo:
        return f"{'-':>10}"
    ok = sum(1 for r in com_rotulo if r.get(campo) == r.get(cv))
    return f"{ok:>3}/{len(com_rotulo):<3} {100 * ok / len(com_rotulo):>3.0f}%"


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

    combate = by_state.get("combat", [])
    if combate:
        print("\nDetalhes de combate (só frames com gabarito):")
        print(f"  cartas na mão   {_acuracia(combate, 'hand_size', 'cv_cards')}")
        print(f"  carta levantada {_acuracia(combate, 'cursor', 'cv_cursor')}")
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
