"""Regressão dos detectores de CV contra frames reais.

Os frames vivem em `frames/`, que é gitignored — num clone novo estes testes são
pulados. A suíte de verdade nasce do dataset rotulado (`python -m src.label`),
que grava em `dataset/` com o estado correto anotado por quem jogou.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.vision.cards import card_bbox, detect_card_slots
from src.vision.minimap import Facing, Turn, read_minimap, relative_turn
from src.vision.screen import Verdict, signature

FRAMES = Path(__file__).resolve().parent.parent / "frames"

# Rotulados à mão inspecionando cada imagem. (arquivo, estado, cursor esperado).
LABELED = [
    ("20260802T154006402_map.png", Verdict.COMBAT, 5),
    ("20260802T154021211_combat_initial.png", Verdict.COMBAT, 5),
    ("20260802T154032207_card_scan_1.png", Verdict.COMBAT, 4),
    ("20260802T154035945_card_scan_2.png", Verdict.COMBAT, 3),
    ("20260802T154039124_card_scan_3.png", Verdict.COMBAT, 2),
    ("20260802T154042627_card_scan_4.png", Verdict.COMBAT, 1),
    ("20260802T154240385_combat_initial.png", Verdict.MAP, None),
    ("20260802T153740534_map.png", Verdict.MAP, None),
    ("20260503T114343443_level_up.png", Verdict.DIALOG, None),
    ("20260802T153731541_combat_initial.png", Verdict.NOT_GAME, None),
    # Achado observando o jogo ao vivo: as cartas do deck também têm círculo de
    # custo, então esta tela passava por combate e o agente tentaria jogar ali.
    ("deck_baralho_referencia.png", Verdict.DECK, None),
]


def load(name: str) -> np.ndarray:
    path = FRAMES / name
    if not path.is_file():
        pytest.skip(f"frame de referência ausente: {name}")
    return cv2.imread(str(path))


@pytest.mark.parametrize(("name", "expected", "_cursor"), LABELED)
def test_classifica_tela(name: str, expected: Verdict, _cursor: int | None):
    assert signature(load(name)).verdict is expected


@pytest.mark.parametrize(
    ("name", "cursor"), [(n, c) for n, v, c in LABELED if v is Verdict.COMBAT]
)
def test_localiza_cursor_em_combate(name: str, cursor: int):
    slots = detect_card_slots(load(name))
    assert slots.selected is not None
    # O índice absoluto pode deslocar se um círculo da ponta estiver encoberto;
    # o que o agente usa é a posição, e ela tem que ser a carta mais destacada.
    assert slots.selected.side >= 29


def test_sem_cartas_no_mapa():
    assert detect_card_slots(load("20260802T154240385_combat_initial.png")).visible_total == 0


def test_bbox_da_carta_cobre_o_circulo():
    slots = detect_card_slots(load("20260802T154032207_card_scan_1.png"))
    circle = slots.selected
    x, y, w, h = card_bbox(circle)
    assert x <= circle.x and y <= circle.y
    assert x + w >= circle.x + circle.w
    assert y + h >= circle.y + circle.h


def test_minimapa_da_posicao_e_direcao():
    minimap = read_minimap(load("20260802T154240385_combat_initial.png"))
    assert minimap is not None
    assert minimap.facing is Facing.EAST  # conferido olhando o frame
    assert minimap.walkable.any()


def test_minimapa_ausente_em_combate():
    assert read_minimap(load("20260802T154006402_map.png")) is None


@pytest.mark.parametrize(
    ("atual", "desejada", "esperado"),
    [
        (Facing.NORTH, Facing.NORTH, Turn.FORWARD),
        (Facing.NORTH, Facing.EAST, Turn.RIGHT),
        (Facing.NORTH, Facing.WEST, Turn.LEFT),
        (Facing.NORTH, Facing.SOUTH, Turn.BACK),
        (Facing.WEST, Facing.NORTH, Turn.RIGHT),
        (Facing.EAST, Facing.NORTH, Turn.LEFT),
        (Facing.SOUTH, Facing.EAST, Turn.LEFT),
    ],
)
def test_aritmetica_de_giro(atual: Facing, desejada: Facing, esperado: Turn):
    assert relative_turn(atual, desejada) is esperado
