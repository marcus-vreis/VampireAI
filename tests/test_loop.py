"""Controle externo do agente: contadores de falha e códigos de saída."""

from __future__ import annotations

from unittest import mock

import pytest

from src import agent
from src.llm import ModelUnavailableError
from src.states import GameNotFocusedError, GameState, NotTheGameError


@pytest.fixture(autouse=True)
def sem_espera():
    """Zera os sleeps: o loop dorme entre passos e ao esperar foco."""
    with (
        mock.patch.object(agent.time, "sleep"),
        mock.patch.object(agent, "gamepad") as pad,
        mock.patch.object(agent, "preflight", return_value=True),
        mock.patch.object(agent, "default_memory") as mem,
    ):
        mem.return_value = mock.MagicMock()
        yield pad


def rodar(efeitos, iters=10):
    with mock.patch.object(agent, "_step", side_effect=efeitos) as passo:
        return agent.loop(max_iters=iters), passo


def test_respeita_o_limite_de_iteracoes():
    codigo, passo = rodar([GameState.MAP] * 10, iters=4)
    assert codigo == 0
    assert passo.call_count == 4


def test_solta_o_gamepad_ao_sair(sem_espera):
    rodar([GameState.MAP], iters=1)
    assert sem_espera.reset.call_count == 1


def test_solta_o_gamepad_mesmo_abortando(sem_espera):
    rodar([NotTheGameError("x")], iters=5)
    assert sem_espera.reset.call_count == 1


def test_captura_fora_do_jogo_aborta():
    codigo, _ = rodar([NotTheGameError("outra janela")], iters=5)
    assert codigo == 2


def test_modelo_indisponivel_aborta_com_codigo_proprio():
    """Insistir não adianta: o servidor responde e o modelo não roda."""
    codigo, _ = rodar([ModelUnavailableError("runner morreu")], iters=5)
    assert codigo == 3


def test_tres_falhas_seguidas_abortam():
    codigo, passo = rodar([RuntimeError("x")] * 5, iters=10)
    assert codigo == 1
    assert passo.call_count == 3


def test_falha_isolada_nao_aborta():
    """O contador zera a cada passo bem-sucedido."""
    efeitos = [RuntimeError("x"), GameState.MAP, RuntimeError("y"), GameState.MAP]
    codigo, passo = rodar(efeitos, iters=4)
    assert codigo == 0
    assert passo.call_count == 4


def test_perder_o_foco_espera_em_vez_de_abortar():
    """Alternar janela ou uma notificação roubando foco é transitório.

    Abortar a run por isso seria desproporcional.
    """
    efeitos = [GameNotFocusedError("sem foco")] * 3 + [GameState.MAP]
    codigo, passo = rodar(efeitos, iters=len(efeitos))
    assert codigo == 0
    assert passo.call_count == 4, "cada espera consome uma iteração, e o jogo volta"


def test_foco_perdido_pra_sempre_acaba_desistindo():
    codigo, passo = rodar(
        [GameNotFocusedError("sem foco")] * 50, iters=agent._MAX_FOCUS_WAITS + 5
    )
    assert codigo == 2
    assert passo.call_count == agent._MAX_FOCUS_WAITS
