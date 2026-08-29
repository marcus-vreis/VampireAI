"""Renderiza o que a CV enxerga num frame, anotado.

Existe porque afinar limiar no escuro custa caro: durante o desenvolvimento deste
módulo, olhar a máscara resolveu em uma tentativa o que várias rodadas de ajuste
de número não tinham resolvido. Use isto antes de mexer em qualquer limiar.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.vision.cards import card_bbox, detect_card_slots
from src.vision.icons import IconKind, find_icons
from src.vision.minimap import read_minimap
from src.vision.screen import signature

_GREEN = (80, 220, 80)
_RED = (60, 60, 240)
_YELLOW = (60, 220, 240)
_CYAN = (240, 220, 60)
_MAGENTA = (220, 80, 220)

_ICON_COLOR = {
    IconKind.ENEMY: _RED,
    IconKind.BOSS: _MAGENTA,
    IconKind.QUESTION: _YELLOW,
}


def _label(img: np.ndarray, text: str, at: tuple[int, int], color) -> None:
    cv2.putText(img, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def _draw_cards(vis: np.ndarray, frame: np.ndarray) -> None:
    slots = detect_card_slots(frame)
    for i, circle in enumerate(slots.circles):
        chosen = i == slots.selected_idx
        color = _YELLOW if chosen else _GREEN
        cv2.rectangle(vis, (circle.x, circle.y), (circle.x + circle.w, circle.y + circle.h), color, 2)
        _label(vis, str(i), (circle.x, circle.y - 4), color)
        if chosen:
            x, y, w, h = card_bbox(circle)
            cv2.rectangle(vis, (max(0, x), max(0, y)), (x + w, y + h), _YELLOW, 2)


def _draw_minimap(vis: np.ndarray, frame: np.ndarray) -> None:
    minimap = read_minimap(frame)
    if minimap is None:
        return
    x0, y0, x1, y1 = minimap.box
    cv2.rectangle(vis, (x0, y0), (x1, y1), _CYAN, 1)

    overlay = vis[y0:y1, x0:x1]
    overlay[minimap.walkable] = (0.55 * overlay[minimap.walkable] + np.array([0, 90, 0])).astype(np.uint8)

    px, py = minimap.player
    cv2.circle(vis, (x0 + px, y0 + py), 5, _CYAN, -1)
    _label(vis, minimap.facing.value, (x0 + px + 8, y0 + py), _CYAN)

    for icon in find_icons(minimap.gray, minimap.arrow_side):
        color = _ICON_COLOR[icon.kind]
        cv2.circle(vis, (x0 + icon.x, y0 + icon.y), 8, color, 2)


def annotate(frame: np.ndarray) -> np.ndarray:
    """Cópia do frame com cartas, cursor, minimapa e ícones marcados."""
    vis = frame.copy()
    sig = signature(frame)
    _draw_cards(vis, frame)
    _draw_minimap(vis, frame)
    _label(
        vis,
        f"{sig.verdict.value}  pergaminho={sig.parchment}  ardosia={sig.slate}  cartas={sig.cards}",
        (14, 24),
        _CYAN,
    )
    return vis


def main() -> int:
    parser = argparse.ArgumentParser(description="Anota um frame com o que a CV vê.")
    parser.add_argument("frame", help="PNG a anotar")
    parser.add_argument("--out", help="Caminho de saída (default: ao lado, com sufixo _debug)")
    args = parser.parse_args()

    frame = cv2.imread(args.frame)
    if frame is None:
        raise SystemExit(f"não consegui ler {args.frame}")

    source = Path(args.frame)
    out = Path(args.out) if args.out else source.with_name(f"{source.stem}_debug.png")
    cv2.imwrite(str(out), annotate(frame))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
