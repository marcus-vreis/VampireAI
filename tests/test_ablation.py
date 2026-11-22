"""Ablação: comparação justa entre CV e o caminho aposentado."""

from __future__ import annotations

import pytest

from src import ablation


def test_gabarito_cobre_todos_os_frames():
    """Rotular com a própria CV seria circular e daria 100% sempre."""
    gabarito = ablation.carregar_gabarito()
    arquivos = {p.name for p in ablation.REFERENCIA.glob("*.png")}
    if not arquivos:
        pytest.skip("frames de referência ausentes")
    assert set(gabarito) == arquivos


def test_traduz_estados_pro_mesmo_vocabulario():
    """A CV agrupa as telas de escolha; o VLM responde o estado exato. Comparar
    exige trazer os dois pro mesmo nível."""
    assert ablation._grupo("level_up") == "dialog"
    assert ablation._grupo("chest") == "dialog"
    assert ablation._grupo("menu") == "outro"
    assert ablation._grupo("combat") == "combat"
    assert ablation._grupo("map") == "map"


def test_identifica_o_que_o_prompt_nao_sabia_dizer():
    """`deck` e `not_game` foram descobertos DEPOIS que o prompt saiu de uso.
    Contar isso como erro do modelo seria trapaça."""
    expressaveis = ablation.opcoes_do_prompt()
    assert "combat" in expressaveis
    assert "map" in expressaveis
    assert "dialog" in expressaveis
    assert "deck" not in expressaveis
    assert "not_game" not in expressaveis


def test_pontua_separando_o_justo_do_impossivel():
    acc = {"acertos": 0, "total": 0, "tempos": [], "erros": [], "just_ok": 0, "just_n": 0}
    ablation._pontuar(acc, "a.png", "combat", "combat", 0.01, justo=True)
    ablation._pontuar(acc, "b.png", "dialog", "deck", 0.01, justo=False)
    assert acc["total"] == 2
    assert acc["acertos"] == 1
    assert acc["just_n"] == 1, "o impossível não entra na conta justa"
    assert "fora do vocabulário" in acc["erros"][0]
