"""Handler de baú. Nunca tinha sido exercitado, nem com mock."""

from __future__ import annotations

from unittest import mock

import pytest

from src import agent
from src.states import GameState


def percebido(state=GameState.CHEST, **data):
    return mock.Mock(state=state, data=data or None)


@pytest.fixture
def executor():
    with (
        mock.patch.object(agent, "grab", return_value="x.png"),
        mock.patch.object(agent, "input_exec") as exec_,
        mock.patch.object(agent, "_decide_choice") as decide,
    ):
        decide.return_value = mock.Mock(indice_alvo=1, motivo="sinergia")
        yield exec_, decide


def test_bau_com_cartas_escolhe_em_vez_de_sacar(executor):
    """A leitura por recorte não preenche `tipo`, e o handler lia isso como
    "vazio" — sacava dinheiro e descartava a recompensa."""
    exec_, decide = executor
    opcoes = [{"posicao": i, "nome": f"c{i}", "e_bonus": False} for i in range(3)]
    with mock.patch.object(agent, "perceive", return_value=percebido(opcoes=opcoes)):
        agent.handle_chest(None)
    assert decide.call_count == 1
    assert exec_.cancel.call_count == 0, "não pode sacar dinheiro havendo carta"
    assert exec_.select_and_confirm.call_count == 1


def test_bau_sem_opcoes_saca_dinheiro(executor):
    exec_, decide = executor
    with mock.patch.object(agent, "perceive", return_value=percebido(opcoes=[])):
        agent.handle_chest(None)
    assert exec_.cancel.call_count == 1
    assert decide.call_count == 0


def test_bau_de_bonus_tambem_escolhe(executor):
    exec_, _ = executor
    opcoes = [{"posicao": 0, "nome": "b", "e_bonus": True}]
    with mock.patch.object(
        agent, "perceive", return_value=percebido(opcoes=opcoes, tipo="bonus")
    ):
        agent.handle_chest(None)
    assert exec_.cancel.call_count == 0


def test_estado_mudou_no_meio_do_handler(executor):
    exec_, decide = executor
    with mock.patch.object(agent, "perceive", return_value=percebido(state=GameState.MAP)):
        agent.handle_chest(None)
    assert decide.call_count == 0
    assert exec_.cancel.call_count == 0


def test_indice_fora_do_intervalo_e_preso(executor):
    """O modelo pode devolver um índice que não existe."""
    exec_, decide = executor
    decide.return_value = mock.Mock(indice_alvo=99, motivo="x")
    opcoes = [{"posicao": i, "nome": f"c{i}", "e_bonus": False} for i in range(2)]
    with mock.patch.object(agent, "perceive", return_value=percebido(opcoes=opcoes)):
        agent.handle_chest(None)
    passos = exec_.select_and_confirm.call_args[0][0]
    assert passos == 1 - 1, "índice 99 preso a 1, cursor em 1 -> zero passos"
