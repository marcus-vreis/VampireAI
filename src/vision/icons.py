"""Ícones do minimapa: inimigos, chefe, interrogação.

Os ícones são sprites fixos desenhados sempre no mesmo tom (cinza 136, contra
194-206 do piso e 160-179 da névoa), então template matching acerta com folga.
Segmentar por cor não bastava: as linhas de borda das salas usam o mesmo tom e
grudam no crânio, e qualquer abertura morfológica forte o bastante pra separá-las
parte o crânio ao meio.

O minimapa muda de zoom entre fases — a seta do jogador mede 16px numa e 19px
noutra — então a busca roda em algumas escalas em torno da referência.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import cache
from pathlib import Path

import cv2
import numpy as np

_TEMPLATE_DIR = Path(__file__).parent / "templates"
# Escala em que os templates foram recortados, medida pela seta do jogador.
_REFERENCE_ARROW = 16
# Varredura ABSOLUTA de escala, passo fino. Derivar a escala do tamanho da seta
# parecia natural mas engana: num frame a seta mede 19px (razão 1.19) enquanto os
# ícones pedem 1.30, e um passo grosso passa direto pelo pico — a correlação cai
# de 0.85 pra 0.60 com 0.05 de erro de escala, porque o sprite é pixel art.
_SCALES = tuple(round(0.80 + 0.05 * i, 2) for i in range(17))
# Limiar por tipo. A interrogação é um template pequeno e de baixo contraste
# (15x9), então casa com ruído do pergaminho em várias escalas; ela não é usada
# pra navegar, então pode ser exigente sem custo.
_MIN_SCORE = {
    "inimigo": 0.70,
    "chefe": 0.70,
    "interrogacao": 0.90,
}
_NMS_RADIUS = 10


class IconKind(str, Enum):
    ENEMY = "inimigo"
    BOSS = "chefe"
    QUESTION = "interrogacao"


_TEMPLATE_FILE = {
    IconKind.ENEMY: "skull.png",
    IconKind.BOSS: "boss.png",
    IconKind.QUESTION: "question.png",
}


@dataclass(frozen=True)
class Icon:
    kind: IconKind
    x: int  # centro, em pixels do minimapa
    y: int
    score: float


@cache
def _template(kind: IconKind) -> np.ndarray:
    path = _TEMPLATE_DIR / _TEMPLATE_FILE[kind]
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"template ausente: {path}")
    return img


def _match_one(
    gray: np.ndarray, tpl: np.ndarray, scale: float, min_score: float
) -> list[tuple[int, int, float]]:
    if scale != 1.0:
        tpl = cv2.resize(
            tpl, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )
    if tpl.shape[0] > gray.shape[0] or tpl.shape[1] > gray.shape[1]:
        return []
    result = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.nonzero(result >= min_score)
    half_h, half_w = tpl.shape[0] // 2, tpl.shape[1] // 2
    return [
        (int(x) + half_w, int(y) + half_h, float(result[y, x]))
        for x, y in zip(xs, ys, strict=True)
    ]


def _suppress(hits: list[Icon]) -> list[Icon]:
    """Mantém só o melhor acerto de cada vizinhança."""
    kept: list[Icon] = []
    for icon in sorted(hits, key=lambda i: -i.score):
        if any(
            abs(icon.x - k.x) < _NMS_RADIUS and abs(icon.y - k.y) < _NMS_RADIUS
            for k in kept
        ):
            continue
        kept.append(icon)
    return kept


def find_icons(minimap_gray: np.ndarray, arrow_side: int = _REFERENCE_ARROW) -> list[Icon]:
    """Ícones no recorte do minimapa em escala de cinza.

    `arrow_side` fica só como registro do zoom observado; a escala é varrida, não
    inferida dele (ver comentário em `_SCALES`).
    """
    hits: list[Icon] = []
    for kind in IconKind:
        tpl = _template(kind)
        threshold = _MIN_SCORE[kind.value]
        for scale in _SCALES:
            for x, y, score in _match_one(minimap_gray, tpl, scale, threshold):
                hits.append(Icon(kind=kind, x=x, y=y, score=score))
    # Chefe e crânio compartilham o desenho da caveira; onde os dois batem, o
    # chefe (template maior, com chifres) tem que vencer.
    ordered = sorted(hits, key=lambda i: (i.kind is not IconKind.BOSS, -i.score))
    return _suppress(ordered)
