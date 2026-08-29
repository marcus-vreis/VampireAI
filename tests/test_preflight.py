"""Verificação de pré-voo antes de uma run."""

from __future__ import annotations

from unittest import mock

import pytest

from src import agent


@pytest.fixture
def sem_espera():
    with mock.patch.object(agent.time, "sleep"):
        yield


def janela(source="win32"):
    return mock.Mock(source=source, rect=mock.Mock(w=1280, h=720, x=320, y=191))


def test_recusa_sem_a_janela_do_jogo(sem_espera):
    """Antes, isso só aparecia na primeira captura."""
    with mock.patch.object(agent, "find_game_window", return_value=janela("config")):
        assert agent.preflight(countdown_s=0) is False


def test_recusa_com_o_modelo_fora(sem_espera):
    """Sem o pré-voo, Ollama fora do ar só aparecia na primeira decisão, depois
    de três tentativas com backoff — vários minutos até a causa ficar visível."""
    with (
        mock.patch.object(agent, "find_game_window", return_value=janela()),
        mock.patch.object(agent, "ask_vlm", side_effect=RuntimeError("connection refused")),
    ):
        assert agent.preflight(countdown_s=0) is False


def test_aprova_com_tudo_no_lugar(sem_espera):
    with (
        mock.patch.object(agent, "find_game_window", return_value=janela()),
        mock.patch.object(agent, "ask_vlm", return_value="ok"),
    ):
        assert agent.preflight(countdown_s=0) is True


def test_da_tempo_de_trocar_de_janela():
    """O boot_delay_s do gamepad (0.5s) serve pro driver inicializar, não pra uma
    pessoa alternar do terminal pro jogo."""
    with (
        mock.patch.object(agent, "find_game_window", return_value=janela()),
        mock.patch.object(agent, "ask_vlm", return_value="ok"),
        mock.patch.object(agent.time, "sleep") as dormiu,
    ):
        agent.preflight(countdown_s=3)
    assert dormiu.call_count == 3


def test_loop_nao_comeca_com_pre_voo_reprovado():
    with (
        mock.patch.object(agent, "preflight", return_value=False),
        mock.patch.object(agent, "gamepad"),
        mock.patch.object(agent, "_step") as passo,
    ):
        assert agent.loop(max_iters=5) == 2
    assert passo.call_count == 0, "não pode dar um passo sequer"
