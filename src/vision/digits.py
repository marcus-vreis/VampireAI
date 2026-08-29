"""Leitura de algarismos do HUD por glifo aprendido.

O problema: a mana é um algarismo grande e nítido sobre o orbe azul — caso
trivial pra qualquer OCR — mas o Tesseract é um binário externo que nem toda
instalação tem, e cair no modelo custa ~830ms por turno.

A solução é a mesma do `CardDB`: **o modelo ensina, a CV assume**. Enquanto um
glifo é desconhecido, quem lê é o modelo (ou o Tesseract), e o resultado é
guardado. Da segunda aparição em diante o glifo é reconhecido em microssegundos.
Como o jogo é pixel art com fonte fixa, o glifo de um "3" é sempre o mesmo mapa
de bits — o reconhecimento é exato, não aproximado.

Um rótulo errado envenenaria o glifo pra sempre, então um algarismo só passa a
ser servido depois de `_MIN_VOTES` observações concordantes.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

# Normalização do glifo antes de virar chave: pixel art escala mal, mas 12x18
# preserva a estrutura dos algarismos e absorve um pixel de diferença no recorte.
_NORM_W, _NORM_H = 12, 18
# Duas observações concordantes antes de confiar. Uma leitura errada do modelo
# envenenaria o glifo permanentemente.
_MIN_VOTES = 2

_MIN_AREA = 60
_MIN_HEIGHT = 14
_MIN_ASPECT, _MAX_ASPECT = 0.20, 1.15
# Densidade separa algarismo de contorno de HUD. No coração, os quatro dígitos
# ficam em 0.52-0.62 e o contorno branco do próprio coração, em 0.09-0.10.
_MIN_DENSITY = 0.30


@dataclass(frozen=True)
class Glyph:
    key: str
    x: int  # ordena algarismos dentro de um número
    y: int  # separa linhas: o coração empilha HP atual sobre HP máximo
    h: int  # altura do algarismo, referência pro corte entre linhas


def _normalize(patch: np.ndarray) -> str:
    small = cv2.resize(patch, (_NORM_W, _NORM_H), interpolation=cv2.INTER_AREA)
    bits = (small > 127).flatten()
    return "".join(f"{b:02x}" for b in np.packbits(bits))


def find_glyphs(mask: np.ndarray) -> list[Glyph]:
    """Algarismos numa máscara binária, da esquerda pra direita.

    Filtra por forma e densidade: algarismo é alto, estreito e cheio. Isso
    descarta o contorno branco do coração, que é uma curva fina e ocupa quase
    toda a caixa.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    found: list[Glyph] = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < _MIN_AREA or h < _MIN_HEIGHT:
            continue
        if not (_MIN_ASPECT <= w / h <= _MAX_ASPECT):
            continue
        if area < _MIN_DENSITY * w * h:
            continue
        patch = (labels[y : y + h, x : x + w] == i).astype(np.uint8) * 255
        found.append(Glyph(key=_normalize(patch), x=x, y=y, h=h))
    found.sort(key=lambda g: g.x)
    return found


class GlyphBook:
    """Glifo → algarismo, com votos. Persistido em JSON."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._votes: dict[str, Counter[str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._votes = {k: Counter(v) for k, v in raw.items()}
        logger.debug("GlyphBook com {} glifos", len(self._votes))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: dict(v) for k, v in self._votes.items()}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def lookup(self, key: str) -> int | None:
        votes = self._votes.get(key)
        if not votes:
            return None
        digit, count = votes.most_common(1)[0]
        return int(digit) if count >= _MIN_VOTES else None

    def observe(self, key: str, digit: int) -> None:
        self._votes.setdefault(key, Counter())[str(digit)] += 1
        self.save()

    def read(self, glyphs: list[Glyph]) -> int | None:
        """Número inteiro, ou None se algum algarismo ainda é desconhecido."""
        if not glyphs:
            return None
        digits = [self.lookup(g.key) for g in glyphs]
        if any(d is None for d in digits):
            return None
        return int("".join(str(d) for d in digits))

    def teach(self, glyphs: list[Glyph], value: int) -> None:
        """Associa um número lido por outra fonte aos glifos que o compõem."""
        text = str(value)
        if len(text) != len(glyphs):
            return  # segmentação e leitura discordam: não dá pra atribuir com certeza
        for glyph, char in zip(glyphs, text, strict=True):
            self.observe(glyph.key, int(char))

    def __len__(self) -> int:
        return sum(1 for v in self._votes.values() if max(v.values()) >= _MIN_VOTES)


def group_rows(glyphs: list[Glyph], gap: float = 0.6) -> list[list[Glyph]]:
    """Agrupa glifos em linhas pela coordenada vertical.

    `gap` é fração da altura MEDIANA dos algarismos: dois glifos separados
    verticalmente por mais que isso estão em linhas diferentes. Usar o
    espalhamento total em vez da altura não funciona — no coração as duas linhas
    ficam a 22px e o espalhamento é 47, o que colapsava tudo numa linha só.
    """
    if not glyphs:
        return []
    ordered = sorted(glyphs, key=lambda g: (g.y, g.x))
    heights = sorted(g.h for g in ordered)
    threshold = max(6.0, gap * heights[len(heights) // 2])
    rows: list[list[Glyph]] = [[ordered[0]]]
    for glyph in ordered[1:]:
        if glyph.y - rows[-1][-1].y > threshold:
            rows.append([glyph])
        else:
            rows[-1].append(glyph)
    return [sorted(row, key=lambda g: g.x) for row in rows]
