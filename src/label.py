"""Sessão de captura rotulada: gera o conjunto de regressão do projeto.

Você joga normalmente. A cada tela interessante, aperta a tecla do estado
correspondente; a ferramenta captura o frame, grava o rótulo e já mostra o que a
CV teria respondido. Divergência aparece na hora, então dá pra caçar os casos
difíceis de propósito em vez de torcer pra aparecerem.

A saída (`dataset/`) é versionável e vira teste automatizado — é o que permite
dizer "contagem 98% em 60 frames" em vez de "parece melhor". É também o conjunto
que a comparação entre modelos vai usar.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
from loguru import logger

from src.capture import grab
from src.config import PROJECT_ROOT
from src.vision.cards import detect_card_slots
from src.vision.minimap import read_minimap
from src.vision.screen import signature

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


def capture_labeled(state: str, hand_size: int | None, cursor: int | None) -> dict:
    """Captura o frame atual, move pro dataset e grava o rótulo."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    shot = grab(state=f"label_{state}")
    target = DATASET_DIR / shot.name
    shot.replace(target)

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "file": target.name,
        "state": state,
        "hand_size": hand_size,
        "cursor": cursor,
        **_observed(target),
    }
    _append(record)
    return record


def _report(record: dict) -> None:
    agree = record["cv_verdict"] in (record["state"], "dialog")
    mark = "ok " if agree else "DIVERGE"
    logger.info(
        "{} rotulado={} cv={} cartas={} cursor={}",
        mark, record["state"], record["cv_verdict"], record["cv_cards"], record["cv_cursor"],
    )


def session(ask_details: bool) -> int:
    print("Sessão de rotulagem. Deixe o jogo visível e volte aqui pra marcar.\n")
    print(_menu())
    total = 0
    while True:
        key = _read_key()
        if key == "q":
            break
        state = _KEYS.get(key)
        if state is None:
            continue
        hand_size = cursor = None
        if ask_details and state == "combat":
            hand_size = _ask_int("quantas cartas na mão? (enter pula) ")
            cursor = _ask_int("índice do cursor, 0 = mais à esquerda? (enter pula) ")
        record = capture_labeled(state, hand_size, cursor)
        _report(record)
        total += 1
    print(f"\n{total} frames gravados em {DATASET_DIR}")
    return 0


def _ask_int(prompt: str) -> int | None:
    raw = input(prompt).strip()
    return int(raw) if raw.isdigit() else None


def summary() -> int:
    if not LABELS_FILE.is_file():
        print("Nenhum rótulo ainda. Rode `python -m src.label` pra criar.")
        return 1
    records = [json.loads(line) for line in LABELS_FILE.read_text(encoding="utf-8").splitlines()]
    by_state: dict[str, list[dict]] = {}
    for r in records:
        by_state.setdefault(r["state"], []).append(r)

    print(f"{len(records)} frames rotulados\n")
    print(f"{'estado':20} {'n':>4} {'CV concorda':>12}")
    for state, rows in sorted(by_state.items()):
        agree = sum(1 for r in rows if r["cv_verdict"] in (state, "dialog"))
        print(f"{state:20} {len(rows):>4} {100 * agree / len(rows):>11.0f}%")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Captura rotulada pro dataset de regressão.")
    parser.add_argument(
        "--summary", action="store_true", help="Resume o dataset já gravado e sai."
    )
    parser.add_argument(
        "--details", action="store_true",
        help="Em combate, também pergunta tamanho da mão e cursor (rótulo mais rico).",
    )
    args = parser.parse_args()

    if args.summary:
        return summary()
    if not sys.stdin.isatty():
        logger.error("Rode num terminal interativo — a rotulagem lê teclas.")
        return 2
    return session(ask_details=args.details)


if __name__ == "__main__":
    raise SystemExit(main())
