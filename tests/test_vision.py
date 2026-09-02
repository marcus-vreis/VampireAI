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

_RAIZ = Path(__file__).resolve().parent.parent
# Gabarito versionado. `frames/` é gitignored E rotacionado durante uma run,
# então teste que dependa dele passa a pular em silêncio — pior que falhar.
FRAMES = _RAIZ / "dataset" / "referencia"

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


@pytest.mark.parametrize(("name", "cursor"), [(n, c) for n, v, c in LABELED if v is Verdict.COMBAT])
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


# --- oclusao da carta levantada -------------------------------------------
# A sequencia 154xxx e UM combate de 6 cartas, capturado a cada passo de uma
# travessia da direita pra esquerda. E o unico gabarito de oclusao que existe:
# `154006402` tem a ultima levantada e mostra as 6, os demais tem uma do meio
# levantada e mostram 5. A diferenca entre eles E o efeito a medir.
_MESMO_COMBATE_DE_6 = {
    "20260802T154006402_map.png": (6, 5, None),  # ultima levantada: nada tapado
    "20260802T154035945_card_scan_2.png": (5, 3, 4),
    "20260802T154039124_card_scan_3.png": (5, 2, 3),
    "20260802T154042627_card_scan_4.png": (5, 1, 2),
}


@pytest.mark.parametrize("nome,esperado", sorted(_MESMO_COMBATE_DE_6.items()))
def test_a_carta_levantada_tapa_exatamente_uma_vizinha(nome, esperado):
    """Num frame so, sem travessia: 5 circulos visiveis viram mao de 6.

    O vao que a oclusao deixa nao passa por espacamento normal — 310-345px
    contra no maximo 146px em qualquer outro vao dos 14 frames de combate do
    dataset/. A folga de 164px e o que permite decidir sem percorrer o leque.
    """
    caminho = FRAMES / nome
    if not caminho.is_file():
        pytest.skip(f"frame ausente: {nome}")
    visiveis, selecionada, tapada = esperado
    slots = detect_card_slots(cv2.imread(str(caminho)))
    assert (slots.visible_total, slots.selected_idx) == (visiveis, selecionada)
    assert slots.hidden_idx == tapada
    assert slots.hand_size == 6, "os quatro frames sao do MESMO combate de 6 cartas"


def test_a_ultima_levantada_e_ambigua_e_nao_finge_que_sabe():
    """`154032207` tem a levantada na ponta direita e mostra 5 circulos, mas a
    mao e de 6 — a sexta esta tapada e nao ha vao depois pra denuncia-la.

    O detector devolve `hidden_idx=None`, que significa "nao sei", e `hand_size`
    continua sendo piso. Fingir 6 aqui seria chutar: o mesmo frame poderia ser
    uma mao de 5 com a ultima levantada. Esse caso exige travessia.
    """
    caminho = FRAMES / "20260802T154032207_card_scan_1.png"
    if not caminho.is_file():
        pytest.skip("frame ausente")
    slots = detect_card_slots(cv2.imread(str(caminho)))
    assert slots.selected_idx == slots.visible_total - 1, "levantada na ponta direita"
    assert slots.hidden_idx is None
    assert slots.hand_size == 5, "piso, nao a verdade (que e 6)"
