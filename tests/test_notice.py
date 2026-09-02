"""Estado `notice`: painel de aviso sem nada pra escolher."""

from __future__ import annotations

from unittest import mock

from src import agent, states
from src.config import PATHS


def test_notice_e_uma_resposta_valida_do_prompt_de_dialogo():
    """Sem esta opção o modelo era forçado a chutar uma das cinco recompensas."""
    assert states.GameState.NOTICE in states._DIALOG_STATES
    prompt = (PATHS.prompts / "detect_dialog.txt").read_text(encoding="utf-8")
    assert '"notice"' in prompt
    assert "Não force uma" in prompt


def test_handler_confirma_o_aviso():
    with mock.patch.object(agent, "input_exec") as exec_:
        agent.handle_notice(None)
    assert exec_.confirm.call_count == 1
    assert exec_.cancel.call_count == 0, "quadrado sacaria dinheiro; o botão do aviso é X"


def test_notice_tem_handler_registrado():
    assert agent._HANDLERS[states.GameState.NOTICE] is agent.handle_notice


def test_notice_tambem_e_saida_do_prompt_de_telas_diversas():
    """Aqui a lista fechada era mais perigosa: forçar uma tela desconhecida em
    `game_over` mata a run, porque `handle_game_over` levanta SystemExit."""
    assert states.GameState.NOTICE in states._OTHER_STATES
    prompt = (PATHS.prompts / "detect_other.txt").read_text(encoding="utf-8")
    assert '"notice"' in prompt
    assert "NÃO chute" in prompt


def test_game_over_encerra_a_run():
    """Confirma o motivo de o prompt ser conservador com game_over."""
    import pytest

    with pytest.raises(SystemExit):
        agent.handle_game_over(None)
