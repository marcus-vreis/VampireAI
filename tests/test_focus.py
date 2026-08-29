"""Pré-condição de foco antes de agir."""

from __future__ import annotations

from unittest import mock

import pytest

from src import agent, gamepad
from src.states import GameNotFocusedError


@pytest.fixture(autouse=True)
def sem_dry_run():
    anterior = gamepad.is_dry_run()
    gamepad.set_dry_run(False)
    yield
    gamepad.set_dry_run(anterior)


def janela(foreground: bool):
    return mock.Mock(foreground=foreground)


def test_recusa_agir_com_o_jogo_atras_de_outra_janela():
    """mss captura uma REGIÃO DA TELA. Já aconteceu de a página da Steam estar
    por cima, ser capturada, passar por combate, e os números de tempo de jogo
    dela virarem leituras de mana de 24 e 8."""
    with (
        mock.patch.object(agent, "find_game_window", return_value=janela(False)),
        pytest.raises(GameNotFocusedError, match="primeiro plano"),
    ):
        agent._require_focus()


def test_prossegue_com_o_jogo_em_primeiro_plano():
    with mock.patch.object(agent, "find_game_window", return_value=janela(True)):
        agent._require_focus()


def test_dry_run_dispensa_a_checagem():
    """No replay os frames vêm de arquivo e nenhum input é emitido — o que está
    na tela agora é irrelevante."""
    gamepad.set_dry_run(True)
    with mock.patch.object(agent, "find_game_window") as procura:
        agent._require_focus()
    assert procura.call_count == 0


def test_o_erro_de_foco_e_um_caso_de_nao_e_o_jogo():
    """O loop já trata NotTheGameError como fatal; foco herda esse tratamento."""
    from src.states import NotTheGameError

    assert issubclass(GameNotFocusedError, NotTheGameError)


def test_cli_avisa_quando_o_jogo_nao_esta_focado():
    """As CLIs emitem input de verdade. A contagem regressiva delas PEDE pra
    focar o jogo, mas pedir não é conferir."""
    from src import window

    with mock.patch.object(window, "find_game_window", return_value=janela(False)):
        assert window.warn_if_unfocused() is False


def test_cli_nao_avisa_com_o_jogo_focado():
    from src import window

    with mock.patch.object(window, "find_game_window", return_value=janela(True)):
        assert window.warn_if_unfocused() is True


def test_cli_avisa_mas_nao_bloqueia():
    """Quem roda uma CLI dessas está depurando de propósito, e às vezes quer ver
    o efeito noutro lugar. Avisar basta; bloquear atrapalharia."""
    from src import window

    with mock.patch.object(window, "find_game_window", return_value=janela(False)):
        window.warn_if_unfocused()  # não levanta


def test_tirar_o_foco_congela_o_agente():
    """Failsafe emergente: alternar pro terminal para o agente na hora.

    A ADR-014 registrou como risco aceito não haver `pyautogui.FAILSAFE` global
    ao adotar o gamepad virtual. A checagem de foco entrou por outro motivo
    (captura pegando a Steam) e resolveu isso de graça.
    """
    from src.states import GameNotFocusedError

    chamadas = []
    with (
        mock.patch.object(agent, "find_game_window", return_value=janela(False)),
        mock.patch.object(agent, "grab", side_effect=lambda *a, **k: chamadas.append(1)),
    ):
        with pytest.raises(GameNotFocusedError):
            agent._require_focus()
    assert chamadas == [], "nem chega a capturar, quanto mais a agir"
