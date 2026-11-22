"""Cercas que o minimapa não desenha: o agente aprende batendo (ADR-104)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import pytest

from src import agent
from src.vision.minimap import Facing, Minimap, read_minimap

_FRAME = (
    Path(__file__).resolve().parent.parent
    / "dataset"
    / "referencia"
    / "20261122T144245137_map_cerca_ao_norte.png"
)


@pytest.fixture
def mapa() -> Minimap:
    if not _FRAME.is_file():
        pytest.skip("frame de referência ausente")
    return read_minimap(cv2.imread(str(_FRAME)))


@pytest.fixture(autouse=True)
def memoria_limpa():
    agent.forget_fences()
    yield
    agent.forget_fences()


def test_andar_sem_sair_da_celula_vira_cerca(mapa):
    agent._LAST_WALK = (mapa.player, Facing.NORTH)
    agent._learn_fence(mapa)
    assert {(mapa.player, Facing.NORTH)} == agent._FENCES


def test_andar_e_mudar_de_celula_nao_vira_cerca(mapa):
    """Passo real é 30px: chegar na vizinha não é bater."""
    x, y = mapa.player
    agent._LAST_WALK = ((x, y + 30), Facing.NORTH)
    agent._learn_fence(mapa)
    assert not agent._FENCES


def test_girar_entre_os_passos_nao_vira_cerca(mapa):
    """Mesma célula olhando pra outro lado: o último comando não foi esse andar."""
    agent._LAST_WALK = (mapa.player, Facing.EAST)
    agent._learn_fence(mapa)
    assert not agent._FENCES


def test_tolera_a_seta_variando_um_pixel(mapa):
    x, y = mapa.player
    agent._LAST_WALK = ((x + 1, y - 2), Facing.NORTH)
    agent._learn_fence(mapa)
    assert len(agent._FENCES) == 1


def test_cerca_some_com_a_mudanca_de_andar(mapa):
    """Coordenada do minimapa só vale no mesmo andar."""
    agent._FENCES.add((mapa.player, Facing.NORTH))
    saida = dataclasses.replace(mapa, on_exit=True)
    agent._forget_fences_if_leaving(saida, agent.Turn.FORWARD)
    assert not agent._FENCES
