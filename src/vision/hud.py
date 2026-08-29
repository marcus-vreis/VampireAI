"""Leitura de mana e HP do HUD.

Três fontes, da mais barata pra mais cara: glifos já aprendidos (microssegundos),
Tesseract (~5ms, se instalado) e o modelo (~830ms). As duas primeiras alimentam o
livro de glifos, então o custo cai sozinho conforme a run avança — ver
`src/vision/digits.py`.
"""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from src.vision.digits import Glyph, GlyphBook, find_glyphs, group_rows

ORB_BOX = (1015, 455, 1145, 585)
HEART_BOX = (140, 450, 275, 580)
_UPSCALE = 4

# O algarismo é branco; o modificador "-1" no canto do orbe é amarelo saturado,
# e os números do coração são brancos sobre vermelho. Exigir saturação baixa e
# brilho alto isola o texto nos dois casos.
_TEXT_LO, _TEXT_HI = np.array([0, 0, 185]), np.array([180, 90, 255])


def text_mask(patch: np.ndarray) -> np.ndarray:
    """Máscara binária do texto claro sobre o fundo colorido do HUD."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, _TEXT_LO, _TEXT_HI)


def orb_glyphs(frame: np.ndarray) -> list[Glyph]:
    """Algarismos do orbe de mana, da esquerda pra direita."""
    x0, y0, x1, y1 = ORB_BOX
    orb = frame[y0:y1, x0:x1]
    if orb.size == 0:
        return []
    return find_glyphs(text_mask(orb))


def read_mana(frame: np.ndarray, book: GlyphBook | None = None) -> int | None:
    """Mana pelo livro de glifos, caindo pro Tesseract. None se ambos falharem.

    Nunca levanta exceção: quem chama decide se vale gastar uma chamada de modelo.
    """
    glyphs = orb_glyphs(frame)
    if book is not None:
        known = book.read(glyphs)
        if known is not None:
            return known

    x0, y0, x1, y1 = ORB_BOX
    digits = _ocr_digits(frame[y0:y1, x0:x1])
    if not digits:
        return None
    value = int(digits[:2])
    if not 0 <= value <= 99:
        return None
    if book is not None:
        book.teach(glyphs, value)
    return value


def read_hp(frame: np.ndarray, book: GlyphBook | None = None) -> tuple[int, int] | None:
    """(hp_atual, hp_max) lidos no coração, ou None se ilegível.

    O coração empilha os dois números, então a separação é por LINHA — agrupar
    glifos por faixa vertical. A versão anterior partia a string de dígitos ao
    meio, o que só funcionava quando o total tinha contagem par e casava 6/1 com
    "61" por acidente.
    """
    if book is None:
        return None
    x0, y0, x1, y1 = HEART_BOX
    heart = frame[y0:y1, x0:x1]
    if heart.size == 0:
        return None
    rows = group_rows(find_glyphs(text_mask(heart)))
    if len(rows) != 2:
        return None
    current, maximum = book.read(rows[0]), book.read(rows[1])
    if current is None or maximum is None:
        return None
    return current, maximum


def _ocr_digits(patch: np.ndarray) -> str:
    try:
        import pytesseract
    except ImportError:
        return ""
    big = cv2.resize(
        text_mask(patch), None, fx=_UPSCALE, fy=_UPSCALE, interpolation=cv2.INTER_CUBIC
    )
    try:
        text = pytesseract.image_to_string(
            cv2.bitwise_not(big),
            config="--psm 7 -c tessedit_char_whitelist=0123456789",
        )
    except Exception as e:  # noqa: BLE001 - tesseract ausente ou mal configurado
        logger.debug("OCR indisponível: {}", e)
        return ""
    return "".join(c for c in text if c.isdigit())
