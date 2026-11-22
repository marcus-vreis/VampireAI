"""Tela escura do próprio jogo não pode abortar a run.

Achado ao vivo em 2026-11-22: o jogo abre num alerta de fotossensibilidade
quase todo preto (brilho 12.8) que pede "pressione para iniciar". A CV classifica
como `not_game` e o agente abortava no passo 0 — mesmo com o sistema tendo
acabado de confirmar que o jogo estava na frente.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pytest

from src import agent
from src.stall import StallDetector
from src.states import GameState, NotTheGameError


@pytest.fixture
def frame_preto(tmp_path):
    path = tmp_path / "boot.png"
    cv2.imwrite(str(path), np.zeros((720, 1280, 3), dtype=np.uint8))
    return path


@pytest.fixture
def jogo_na_frente(frame_preto):
    with (
        mock.patch.object(agent, "grab", return_value=frame_preto),
        mock.patch.object(agent, "find_game_window") as janela,
        mock.patch.object(agent.gamepad, "is_dry_run", return_value=False),
        mock.patch.dict(agent._NUDGE_ACTION, {k: mock.Mock() for k in agent._NUDGE_ACTION}),
    ):
        janela.return_value = SimpleNamespace(foreground=True)
        yield agent._NUDGE_ACTION


def test_tela_escura_com_jogo_na_frente_nao_aborta(jogo_na_frente):
    memory = mock.MagicMock()
    estado = agent._step(memory, GameState.TITLE, StallDetector())
    assert estado is GameState.TITLE, "sem estado novo, o passo mantém o anterior"


def test_tela_escura_parada_e_destravada_com_x(jogo_na_frente):
    """O alerta de boot sai com X; quem aperta é o antitravamento genérico."""
    memory = mock.MagicMock()
    detector = StallDetector()
    for _ in range(detector.patience + 1):
        agent._step(memory, None, detector)
    confirm = jogo_na_frente[agent.Nudge.CONFIRM]
    assert confirm.call_count == 1


def test_sem_confirmacao_do_sistema_continua_abortando(frame_preto):
    """No replay os frames vêm de arquivo: aí `not_game` é mesmo outra coisa."""
    with (
        mock.patch.object(agent, "grab", return_value=frame_preto),
        mock.patch.object(agent.gamepad, "is_dry_run", return_value=True),
        pytest.raises(NotTheGameError),
    ):
        agent._step(mock.MagicMock(), None, StallDetector())
