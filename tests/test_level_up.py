"""Handler de level up e a navegação das telas de escolha."""

from __future__ import annotations

from unittest import mock

import pytest

from src import agent
from src.states import GameState


def percebido(state=GameState.LEVEL_UP, **data):
    return mock.Mock(state=state, data=data or None)


def opcoes(n: int) -> list[dict]:
    return [{"posicao": i, "nome": f"carta{i}", "e_bonus": False} for i in range(n)]


@pytest.fixture
def executor():
    with (
        mock.patch.object(agent, "grab", return_value="x.png"),
        mock.patch.object(agent, "input_exec") as exec_,
        mock.patch.object(agent, "_decide_choice") as decide,
    ):
        decide.return_value = mock.Mock(indice_alvo=2, motivo="sinergia")
        yield exec_, decide


def test_navega_do_cursor_ate_a_escolha(executor):
    exec_, _ = executor
    with mock.patch.object(
        agent, "perceive", return_value=percebido(opcoes=opcoes(4), indice_selecionada=0)
    ):
        agent.handle_level_up(None)
    assert exec_.select_and_confirm.call_args[0][0] == 2, "de 0 até 2: dois passos"


def test_cursor_ilegivel_nao_chuta(executor):
    """O palpite anterior era "está na opção mais à direita", herdado da mão de
    combate. Nas telas de escolha a selecionada é a que sobe — medida no índice 0
    em três frames reais. Errar aqui escolhe a recompensa errada pro resto da run."""
    exec_, decide = executor
    with mock.patch.object(
        agent, "perceive", return_value=percebido(opcoes=opcoes(3), indice_selecionada=None)
    ):
        agent.handle_level_up(None)
    assert exec_.select_and_confirm.call_count == 0
    assert exec_.confirm.call_count == 0


def test_sem_opcoes_confirma_a_destacada(executor):
    exec_, decide = executor
    with mock.patch.object(agent, "perceive", return_value=percebido(opcoes=[])):
        agent.handle_level_up(None)
    assert exec_.confirm.call_count == 1
    assert decide.call_count == 0


def test_quatro_opcoes_sao_suportadas(executor):
    """jogo.md: "Aumente sua Sorte para uma chance de ter 4 escolhas"."""
    exec_, decide = executor
    decide.return_value = mock.Mock(indice_alvo=3, motivo="x")
    with mock.patch.object(
        agent, "perceive", return_value=percebido(opcoes=opcoes(4), indice_selecionada=0)
    ):
        agent.handle_level_up(None)
    assert exec_.select_and_confirm.call_args[0][0] == 3


def test_estado_mudou_no_meio_do_handler(executor):
    exec_, decide = executor
    with mock.patch.object(agent, "perceive", return_value=percebido(state=GameState.COMBAT)):
        agent.handle_level_up(None)
    assert decide.call_count == 0
