"""Leitura das telas de escolha recortando carta por carta."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from src.vision.cards import card_bbox, detect_choice_slots

FRAMES = Path(__file__).resolve().parent.parent / "frames"

# Gabarito conferido olhando cada imagem. O terceiro tem QUATRO opções — jogo.md
# menciona "chance de ter 4 escolhas" — e a última é carta de bônus.
GABARITO = [
    ("20260503T114343443_level_up.png", 3, 0),
    ("20260531T160257173_level_up.png", 3, 0),
    ("20260802T154129487_level_up.png", 4, 0),
]


def load(name: str):
    path = FRAMES / name
    if not path.is_file():
        pytest.skip(f"frame ausente: {name}")
    return cv2.imread(str(path))


@pytest.mark.parametrize(("nome", "total", "selecionada"), GABARITO)
def test_conta_opcoes_e_acha_a_selecionada(nome: str, total: int, selecionada: int):
    slots = detect_choice_slots(load(nome))
    assert slots.visible_total == total
    assert slots.selected_idx == selecionada


def test_selecao_por_altura_nao_cai_no_orbe_do_bonus():
    """A carta de bônus traz um orbe decorativo maior que o círculo de custo.

    Pelo critério de tamanho, ela vencia e o agente escolheria a opção errada.
    A selecionada é a que SOBE: medido, 24-30px acima das demais.
    """
    slots = detect_choice_slots(load("20260802T154129487_level_up.png"))
    maior = max(range(len(slots.circles)), key=lambda i: slots.circles[i].side)
    assert maior == 3, "o orbe do bônus é mesmo o maior círculo"
    assert slots.selected_idx == 0, "mas a selecionada é a mais alta"


def test_recorte_do_bonus_nao_infla_com_o_orbe():
    """Escalar pelo orbe dava 296x368 contra ~180x230 das cartas normais, e o
    excesso de contexto atrapalhava a leitura."""
    from statistics import median

    slots = detect_choice_slots(load("20260802T154129487_level_up.png"))
    lado = int(median(c.side for c in slots.circles))
    larguras = [card_bbox(c, lado)[2] for c in slots.circles]
    assert max(larguras) - min(larguras) <= 1, "todas as cartas recortadas na mesma escala"
