"""Resumo da run impresso ao final."""

from __future__ import annotations

from unittest import mock

import pytest

from src import agent
from src.states import GameState, NotTheGameError


@pytest.fixture(autouse=True)
def resumo_limpo():
    anterior = agent._RUN
    agent._RUN = agent.RunSummary()
    yield agent._RUN
    agent._RUN = anterior


def test_conta_o_que_importa(resumo_limpo):
    r = resumo_limpo
    r.passos = 40
    r.estados.update(["map"] * 30 + ["combat"] * 10)
    r.cartas_jogadas = ["Tomo Vazio", "Otto"]
    r.turnos_encerrados = 2
    texto = r.render()
    assert "40 passos" in texto
    assert "cartas jogadas:      2" in texto
    assert "'map': 30" in texto


def test_destaca_jogada_ilegal(resumo_limpo):
    """A taxa de jogada ilegal é a medida mais direta da qualidade do modelo."""
    resumo_limpo.jogadas_ilegais = 3
    assert "o modelo erra a mana" in resumo_limpo.render()


def test_destaca_destravamento(resumo_limpo):
    """Destravamento significa tela que nenhum handler cobre."""
    resumo_limpo.destravamentos = 2
    assert "telas sem handler" in resumo_limpo.render()


def test_run_limpa_nao_alarma(resumo_limpo):
    texto = resumo_limpo.render()
    assert "o modelo erra" not in texto
    assert "telas sem handler" not in texto


def test_imprime_mesmo_abortando(capsys):
    """Uma run que aborta é justamente quando mais se quer saber o que houve."""
    with (
        mock.patch.object(agent.time, "sleep"),
        mock.patch.object(agent, "gamepad"),
        mock.patch.object(agent, "preflight", return_value=True),
        mock.patch.object(agent, "default_memory", return_value=mock.MagicMock()),
        mock.patch.object(agent, "_step", side_effect=NotTheGameError("x")),
    ):
        assert agent.loop(max_iters=3) == 2
    assert "RESUMO DA RUN" in capsys.readouterr().out


def test_conta_passos_e_estados_no_loop(capsys):
    with (
        mock.patch.object(agent.time, "sleep"),
        mock.patch.object(agent, "gamepad"),
        mock.patch.object(agent, "preflight", return_value=True),
        mock.patch.object(agent, "default_memory", return_value=mock.MagicMock()),
        mock.patch.object(agent, "_step", return_value=GameState.MAP),
    ):
        agent.loop(max_iters=3)
    assert "RESUMO DA RUN" in capsys.readouterr().out
