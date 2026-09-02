"""Detecção de cartas na mão pelo círculo de custo.

Cada carta desenha no canto superior esquerdo um círculo com o custo de mana.
É o único elemento que aparece uma vez por carta, tem tamanho fixo e contraste
alto contra qualquer arte — logo é o marcador natural pra contar e localizar.

Dois fatos do jogo moldam o detector:

1. O círculo da carta SELECIONADA pulsa entre azul e magenta, e fica ~50% maior.
   Por isso a máscara cobre as duas matizes e a seleção é decidida por tamanho,
   não por cor.
2. A carta selecionada sobe e **cobre o círculo da vizinha à direita**, então
   `visible_total` é um piso. Mas o vão que a oclusão deixa é grande demais pra
   passar por espaçamento normal — 310-345px contra no máximo 146px, medido em
   14 frames de combate — e por isso `hand_size` corrige a contagem NUM FRAME
   SÓ, sem travessia. Sobra um caso: quando a levantada é a última visível não
   existe vão depois pra denunciar, e esse ainda precisa de um passo de
   travessia (`src.vision.hand.traverse_hand`).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.vision.digits import GlyphBook, find_glyphs
from src.vision.hud import text_mask

# Faixas HSV (OpenCV: H 0-179) amostradas em frames reais.
_BLUE_LO, _BLUE_HI = np.array([100, 150, 60]), np.array([125, 255, 230])
_MAGENTA_LO, _MAGENTA_HI = np.array([138, 90, 90]), np.array([176, 255, 255])

# Região do leque, com folga à esquerda pra não cortar o círculo da carta da ponta.
_HAND_BOX = (235, 380, 1035, 720)
# Painel central das telas de escolha (level up, baú). As cartas ali usam o mesmo
# círculo de custo da mão, então o mesmo detector serve.
_CHOICE_BOX = (270, 150, 1030, 520)
# Na tela de escolha a selecionada é a mais ALTA, não a maior: cartas de bônus
# trazem um orbe decorativo grande que engana o critério de tamanho. Medido em
# frames reais: a selecionada fica 24-30px acima das outras.
_CHOICE_RAISED_PX = 15

_MIN_SIDE, _MAX_SIDE = 16, 46
_MIN_ASPECT, _MAX_ASPECT = 0.72, 1.38
_MIN_FILL = 0.36
_MIN_BRIGHT_CORE = 0.12
_BRIGHT_V = 190
_DEDUP_X = 26

# A carta selecionada é ~1.4x o círculo normal. Fica no meio das duas escalas.
_SELECTED_SIDE_RATIO = 1.25

# Vao que denuncia um circulo tapado pela carta levantada. Medido nos 14 frames
# de combate do dataset/: 310-345px nos quatro casos de oclusao, contra no
# maximo 146px em qualquer outro vao. 220 fica no meio da folga de 164px.
_OCCLUSION_GAP_PX = 220

# Tamanho da carta em múltiplos do lado do círculo, medido em frames reais.
_CARD_W_RATIO, _CARD_H_RATIO = 7.4, 9.2
_CARD_DX_RATIO, _CARD_DY_RATIO = -0.60, -0.25


@dataclass(frozen=True)
class CostCircle:
    x: int
    y: int
    w: int
    h: int

    @property
    def side(self) -> int:
        return max(self.w, self.h)


@dataclass(frozen=True)
class CardSlots:
    """Círculos visíveis num frame. Ver docstring do módulo sobre oclusão."""

    circles: list[CostCircle]
    selected_idx: int | None

    @property
    def visible_total(self) -> int:
        """Piso do tamanho da mão — a selecionada pode estar cobrindo uma."""
        return len(self.circles)

    @property
    def hidden_idx(self) -> int | None:
        """Posição REAL do círculo tapado pela carta levantada, ou None.

        A carta selecionada sobe e cobre o círculo da vizinha à DIREITA, então
        um frame com carta levantada mostra uma a menos. Isso não precisa da
        travessia pra ser detectado: o vão deixado é grande demais pra ser
        confundido com espaçamento normal.

        Medido nos 14 frames de combate do `dataset/`: o vão entre o círculo
        levantado e o próximo visível deu 310, 321, 333 e 345px nos quatro casos
        de oclusão, contra **no máximo 146px** em qualquer outro vão de qualquer
        frame, com ou sem carta levantada. A folga de 164px é o que permite
        decidir num frame só.

        **Fica um caso ambíguo, de propósito:** quando a levantada é a última
        VISÍVEL, não dá pra saber se ela é a última da mão ou se está cobrindo
        mais uma à direita — o vão que denunciaria não existe, porque não há
        círculo depois. Esse caso continua exigindo travessia; devolver None
        aqui é dizer "não sei", não "não tem".
        """
        i = self.selected_idx
        if i is None or i >= len(self.circles) - 1:
            return None
        vao = self.circles[i + 1].x - self.circles[i].x
        return i + 1 if vao >= _OCCLUSION_GAP_PX else None

    @property
    def hand_size(self) -> int:
        """Tamanho da mão já corrigido pela oclusão.

        Diferente de `visible_total`, que é piso. Continua sendo piso no caso
        ambíguo descrito em `hidden_idx`.
        """
        return len(self.circles) + (1 if self.hidden_idx is not None else 0)

    @property
    def selected(self) -> CostCircle | None:
        if self.selected_idx is None:
            return None
        return self.circles[self.selected_idx]


def _mask(hsv: np.ndarray) -> np.ndarray:
    blue = cv2.inRange(hsv, _BLUE_LO, _BLUE_HI)
    magenta = cv2.inRange(hsv, _MAGENTA_LO, _MAGENTA_HI)
    return cv2.morphologyEx(blue | magenta, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def _bright_core_ratio(hsv: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Fração de pixels claros no centro — é o algarismo do custo.

    Discrimina círculo de texto azul da descrição, que cai na mesma faixa de
    matiz mas não tem núcleo claro. Testa só brilho: no pulso de destaque o
    algarismo fica alaranjado em vez de branco.
    """
    cx, cy = x + w // 2, y + h // 2
    k = max(2, w // 4)
    core = hsv[max(0, cy - k) : cy + k + 1, max(0, cx - k) : cx + k + 1]
    if core.size == 0:
        return 0.0
    return float((core[..., 2] > _BRIGHT_V).mean())


def _candidates(hsv_box: np.ndarray, ox: int, oy: int) -> list[CostCircle]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(_mask(hsv_box), 8)
    found: list[CostCircle] = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if not (_MIN_SIDE <= w <= _MAX_SIDE and _MIN_SIDE <= h <= _MAX_SIDE):
            continue
        if not (_MIN_ASPECT <= w / h <= _MAX_ASPECT):
            continue
        if area < _MIN_FILL * w * h:
            continue
        if _bright_core_ratio(hsv_box, x, y, w, h) < _MIN_BRIGHT_CORE:
            continue
        found.append(CostCircle(x=x + ox, y=y + oy, w=w, h=h))
    found.sort(key=lambda c: c.x)
    return found


def _dedup(circles: list[CostCircle]) -> list[CostCircle]:
    out: list[CostCircle] = []
    for c in circles:
        if out and c.x - out[-1].x < _DEDUP_X:
            if c.w * c.h > out[-1].w * out[-1].h:
                out[-1] = c
            continue
        out.append(c)
    return out


def _selected_index(circles: list[CostCircle]) -> int | None:
    """Índice do círculo destacado, ou None se nenhum se sobressai."""
    if not circles:
        return None
    sides = [c.side for c in circles]
    biggest = max(range(len(sides)), key=lambda i: sides[i])
    others = [s for i, s in enumerate(sides) if i != biggest]
    if not others:
        return biggest
    if sides[biggest] < _SELECTED_SIDE_RATIO * (sum(others) / len(others)):
        return None
    return biggest


def detect_card_slots(frame: np.ndarray) -> CardSlots:
    """Localiza os círculos de custo e o cursor na MÃO de combate."""
    return _slots(frame, _HAND_BOX, _selected_index)


def detect_choice_slots(frame: np.ndarray) -> CardSlots:
    """Idem, no painel central das telas de escolha (level up, baú).

    A seleção é decidida por ALTURA, não por tamanho: cartas de bônus trazem um
    orbe decorativo grande que vencia o critério de tamanho e apontava a carta
    errada. A selecionada é a que sobe — medido em 24-30px acima das demais.
    """
    return _slots(frame, _CHOICE_BOX, _highest_index)


def _slots(frame: np.ndarray, box: tuple[int, int, int, int], pick) -> CardSlots:
    x0, y0, x1, y1 = box
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[y0:y1, x0:x1]
    circles = _dedup(_candidates(hsv, x0, y0))
    return CardSlots(circles=circles, selected_idx=pick(circles))


def _highest_index(circles: list[CostCircle]) -> int | None:
    """Índice do círculo visivelmente mais alto, ou None se estão todos no mesmo nível."""
    if not circles:
        return None
    top = min(range(len(circles)), key=lambda i: circles[i].y)
    others = [c.y for i, c in enumerate(circles) if i != top]
    if not others:
        return top
    return top if min(others) - circles[top].y >= _CHOICE_RAISED_PX else None


def card_bbox(circle: CostCircle, side: int | None = None) -> tuple[int, int, int, int]:
    """Retângulo (x, y, w, h) da carta a que o círculo pertence.

    Proporções medidas em frames reais: a carta é ~7.4x o lado do círculo em
    largura e ~9.2x em altura, ancorada logo acima e à esquerda dele. Com folga
    deliberada — este recorte é o que vai pro VLM ler, então cortar o fim do nome
    ou da descrição custa mais caro que incluir um pedaço da carta vizinha.

    `side` sobrescreve a escala. Serve pras cartas de bônus, que no lugar do
    círculo de custo trazem um orbe decorativo bem maior: usar o lado dele
    inflava o recorte a 296x368 contra os ~180x230 das cartas normais, e o
    excesso de contexto atrapalhava a leitura.
    """
    s = side if side is not None else circle.side
    return (
        int(circle.x + _CARD_DX_RATIO * s),
        int(circle.y + _CARD_DY_RATIO * s),
        int(_CARD_W_RATIO * s),
        int(_CARD_H_RATIO * s),
    )


# Recuo pra dentro do círculo antes de procurar o algarismo, em fração do lado.
# O anel do círculo é uma curva grossa e entraria como componente na máscara.
_COST_INSET = 6
# O algarismo tem ~10px no frame original; find_glyphs pede altura mínima maior
# que isso. Ampliar é mais barato que afrouxar o filtro de forma, que existe pra
# descartar o contorno do coração.
_COST_UPSCALE = 4


def read_costs(
    frame: np.ndarray, slots: CardSlots, book: GlyphBook | None = None
) -> list[int | None]:
    """Custo de mana de cada carta, lido do próprio círculo. None onde ilegível.

    O custo é um algarismo dentro de um círculo que a CV já localiza — pela regra
    da ADR-022 isso é geometria, não semântica, e não devia depender do modelo.
    Hoje ele chega pelo `CardDB`, que é preenchido pelo VLM: a leitura fica presa
    à primeira aparição da carta e ao acerto do modelo naquela chamada.

    A segmentação não precisou de ajuste nenhum — `text_mask` já separa o
    algarismo do azul do círculo. O que falta é vocabulário: os algarismos do
    círculo são um desenho diferente dos do HUD, então o livro de glifos precisa
    aprendê-los (ADR-033), e até lá devolve None. Medido no frame de combate do
    `dataset/`: 1, 1, 1 e 2 saem certos por vizinhança; os dois `0` (que têm um
    corte diagonal) não estão no livro.
    """
    if book is None:
        return [None] * slots.hand_size
    lidos: list[int | None] = [_read_one_cost(frame, c, book) for c in slots.circles]
    oculto = slots.hidden_idx
    if oculto is not None:
        lidos.insert(oculto, None)
    return lidos


def _read_one_cost(frame: np.ndarray, circle: CostCircle, book: GlyphBook) -> int | None:
    m = max(2, circle.side // _COST_INSET)
    patch = frame[circle.y + m : circle.y + circle.h - m, circle.x + m : circle.x + circle.w - m]
    if patch.size == 0:
        return None
    big = cv2.resize(
        patch, None, fx=_COST_UPSCALE, fy=_COST_UPSCALE, interpolation=cv2.INTER_NEAREST
    )
    return book.read(find_glyphs(text_mask(big)))
