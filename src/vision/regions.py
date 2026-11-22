"""Regiões de UI normalizadas ao client area de 1280x720.

Absolutas eram frágeis: a captura escorregava entre sessões (barra de título
presente ou não), então os crops saíam do lugar. Com `src.window` capturando o
client area real, estas coordenadas voltam a ser estáveis; guardamos em fração
para sobreviver a mudança de resolução.
"""

from __future__ import annotations

from dataclasses import dataclass

REFERENCE_W = 1280
REFERENCE_H = 720


@dataclass(frozen=True)
class Region:
    """Retângulo em fração do client area. `to_px` resolve pro tamanho real."""

    left: float
    top: float
    right: float
    bottom: float

    def to_px(self, width: int = REFERENCE_W, height: int = REFERENCE_H) -> tuple[int, int, int, int]:
        return (
            int(self.left * width),
            int(self.top * height),
            int(self.right * width),
            int(self.bottom * height),
        )


def _from_px(left: int, top: int, right: int, bottom: int) -> Region:
    return Region(left / REFERENCE_W, top / REFERENCE_H, right / REFERENCE_W, bottom / REFERENCE_H)


# Medidos em frames reais de 1280x720. O leque vai muito além do crop antigo
# (380,460,480,260), que cortava a carta da ponta esquerda e fatiava a
# selecionada ao meio — causa raiz da contagem e leitura erradas.
HAND = _from_px(250, 380, 1030, 720)
MINIMAP = _from_px(705, 512, 1010, 720)
MANA_ORB = _from_px(1020, 460, 1140, 580)
HP_HEART = _from_px(150, 455, 265, 570)
VIEWPORT = _from_px(258, 28, 1022, 512)
CHOICE_AREA = _from_px(280, 140, 1010, 560)

ALL: dict[str, Region] = {
    "hand": HAND,
    "minimap": MINIMAP,
    "mana_orb": MANA_ORB,
    "hp_heart": HP_HEART,
    "viewport": VIEWPORT,
    "choice_area": CHOICE_AREA,
}
