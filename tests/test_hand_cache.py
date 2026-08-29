"""Reaproveitamento da mão entre jogadas do mesmo turno."""

from __future__ import annotations

from unittest import mock

import pytest

from src import agent
from src.perception import HandScan
from src.schemas import CardScanFrame


def carta(nome: str, mana: int = 1) -> CardScanFrame:
    return CardScanFrame(nome=nome, mana=mana, descricao=None, tipo="ataque")


@pytest.fixture(autouse=True)
def limpa():
    agent.forget_hand()
    yield
    agent.forget_hand()


@pytest.fixture
def mao():
    return HandScan(
        cards=[carta("Tomo", 0), carta("Otto", 2), carta("Faca", 1)],
        cursor_idx=2,
        mana=3,
        hp=(61, 61),
    )


def test_primeira_entrada_percorre_a_mao(mao):
    with (
        mock.patch.object(agent, "scan_combat_hand", return_value=mao) as scan,
        mock.patch.object(agent, "read_hud") as hud,
    ):
        assert agent._current_hand().cards == mao.cards
    assert scan.call_count == 1
    assert hud.call_count == 0


def test_entrada_seguinte_reusa_e_so_atualiza_o_hud(mao):
    """Refazer a travessia custava ~2.5s e não acrescentava informação."""
    agent._HAND = mao
    with (
        mock.patch.object(agent, "scan_combat_hand") as scan,
        mock.patch.object(agent, "read_hud", return_value=(1, (40, 61))) as hud,
    ):
        atual = agent._current_hand()
    assert scan.call_count == 0
    assert hud.call_count == 1
    assert atual.cards == mao.cards
    assert atual.mana == 1, "a mana tem que vir fresca, não do cache"
    assert atual.hp == (40, 61)


def test_jogar_remove_a_carta_da_mao_conhecida(mao):
    with (
        mock.patch.object(agent, "seek_card", return_value=True),
        mock.patch.object(agent, "input_exec"),
        mock.patch.object(agent, "default_carddb"),
    ):
        agent._play_card(1, mao, None, "teste")
    assert [c.nome for c in agent._HAND.cards] == ["Tomo", "Faca"]


def test_falha_ao_posicionar_esquece_a_mao(mao):
    """Se o cursor não chegou na carta, a mão conhecida pode estar errada."""
    agent._HAND = mao
    with (
        mock.patch.object(agent, "seek_card", return_value=False),
        mock.patch.object(agent, "input_exec") as exec_,
        mock.patch.object(agent, "default_carddb"),
    ):
        agent._play_card(1, mao, None, "teste")
    assert agent._HAND is None
    assert exec_.confirm.call_count == 0, "não pode confirmar sem ter chegado na carta"


def test_transicao_de_estado_esquece_a_mao(mao):
    agent._HAND = mao
    with (
        mock.patch.object(agent, "grab", return_value="x.png"),
        mock.patch.object(agent, "perceive") as perceive,
        mock.patch.object(agent, "cv2"),
        mock.patch.object(agent, "_HANDLERS", {}),
        mock.patch.object(agent, "_require_focus"),
    ):
        perceive.return_value = mock.Mock(state=agent.GameState.MAP)
        detector = mock.Mock(stuck=False)
        agent._step(agent.default_memory(), agent.GameState.COMBAT, detector)
    assert agent._HAND is None
