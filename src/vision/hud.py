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

# O algarismo é acinzentado/branco; o modificador "-1" no canto do orbe é amarelo
# saturado, e os números do coração são claros sobre vermelho. Saturação baixa
# elimina os dois fundos.
_MAX_SATURATION = 90
# Margem acima do limiar de Otsu, em fração da distância até o topo do brilho.
# Otsu sozinho deixava o "6" do coração grudar na barra de fração. Faixa segura
# medida (4 casos, duas iluminações): 0.45 a 0.60. 0.55 fica no meio.
_OTSU_MARGIN = 0.55
_BRIGHT_PERCENTILE = 99.5


def text_mask(patch: np.ndarray) -> np.ndarray:
    """Máscara binária do texto claro sobre o fundo colorido do HUD.

    O limiar é ADAPTATIVO, não fixo: o jogo escurece o HUD inteiro quando um
    painel está aberto, e um corte absoluto em brilho perdia todos os dígitos do
    coração nessa condição. Otsu acha a separação natural entre texto e fundo em
    qualquer iluminação; a margem acima dele evita que o algarismo grude em
    elementos vizinhos como a barra de fração.
    """
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    value = hsv[..., 2].copy()
    value[hsv[..., 1] >= _MAX_SATURATION] = 0  # só concorre o que é acinzentado
    otsu, _ = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    top = float(np.percentile(value, _BRIGHT_PERCENTILE))
    threshold = otsu + (top - otsu) * _OTSU_MARGIN
    return ((value > threshold) * 255).astype(np.uint8)


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


def heart_rows(frame: np.ndarray) -> list[list[Glyph]]:
    """Glifos do coração agrupados em linhas: HP atual em cima, máximo embaixo."""
    x0, y0, x1, y1 = HEART_BOX
    heart = frame[y0:y1, x0:x1]
    if heart.size == 0:
        return []
    return group_rows(find_glyphs(text_mask(heart)))


def read_hp(frame: np.ndarray, book: GlyphBook | None = None) -> tuple[int, int] | None:
    """(hp_atual, hp_max) lidos no coração, ou None se ilegível.

    O coração empilha os dois números, então a separação é por LINHA — agrupar
    glifos por faixa vertical. A versão anterior partia a string de dígitos ao
    meio, o que só funcionava quando o total tinha contagem par e casava 6/1 com
    "61" por acidente.

    **As duas primeiras linhas, não "exatamente duas".** Em combate o jogo
    desenha mais um indicador dentro de `HEART_BOX`, embaixo e à esquerda do
    coração; exigir duas linhas fazia o HP falhar justamente no único estado em
    que ele importa pra decisão. Medido nos dois frames do `dataset/`: o par de
    HP fica em y=31..71, x=51..65, altura 17-18; o intruso em y=71..93, x=26,
    altura 22 — abaixo do par, que por isso continua sendo o topo.
    """
    if book is None:
        return None
    rows = heart_rows(frame)
    if len(rows) < 2:
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
