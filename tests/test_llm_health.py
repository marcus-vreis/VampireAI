"""Falha rápida quando o runner do modelo morre."""

from __future__ import annotations

from unittest import mock

import pytest

from src import llm
from src.llm import ModelUnavailableError, _looks_like_dead_runner


@pytest.mark.parametrize(
    ("mensagem", "esperado"),
    [
        ("model runner has unexpectedly stopped, this may be due to resource limitations", True),
        ("GGML_ASSERT(a->ne[2] * 4 == b->ne[0]) failed", True),
        ("failed to load model", True),
        ("Connection refused", False),
        ("timeout", False),
    ],
)
def test_reconhece_runner_morto(mensagem: str, esperado: bool):
    assert _looks_like_dead_runner(Exception(mensagem)) is esperado


def test_desiste_de_imediato_quando_a_sonda_tambem_falha():
    """Observado numa execução real: 3 tentativas de ~42s por chamada, mais de
    2 minutos por passo, com o agente parecendo travado."""
    erro = Exception("model runner has unexpectedly stopped")
    with (
        mock.patch.object(llm, "_get_client") as cliente,
        mock.patch.object(llm, "_runner_alive", return_value=False) as sonda,
    ):
        cliente.return_value.chat.completions.create.side_effect = erro
        with pytest.raises(ModelUnavailableError, match="runner morreu"):
            llm.ask_vlm(None, "oi")
    assert cliente.return_value.chat.completions.create.call_count == 1
    assert sonda.call_count == 1


def test_continua_tentando_se_a_sonda_responde():
    """Se o servidor está vivo, o erro foi daquela chamada — vale repetir."""
    erro = Exception("model runner has unexpectedly stopped")
    with (
        mock.patch.object(llm, "_get_client") as cliente,
        mock.patch.object(llm, "_runner_alive", return_value=True),
        mock.patch.object(llm.time, "sleep"),
    ):
        cliente.return_value.chat.completions.create.side_effect = erro
        with pytest.raises(RuntimeError) as exc:
            llm.ask_vlm(None, "oi")
    assert not isinstance(exc.value, ModelUnavailableError)
    assert cliente.return_value.chat.completions.create.call_count == llm.LLM.max_retries


def test_erro_comum_de_rede_nao_dispara_a_sonda():
    with (
        mock.patch.object(llm, "_get_client") as cliente,
        mock.patch.object(llm, "_runner_alive") as sonda,
        mock.patch.object(llm.time, "sleep"),
    ):
        cliente.return_value.chat.completions.create.side_effect = Exception("Connection refused")
        with pytest.raises(RuntimeError):
            llm.ask_vlm(None, "oi")
    assert sonda.call_count == 0
