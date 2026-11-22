"""Validação de jogada e regra de reserva do combate."""

from __future__ import annotations

import pytest

from src.combat import Rejection, affordable, fallback_index, validate
from src.schemas import CardScanFrame


def card(nome: str, mana: int | None, tipo: str = "ataque") -> CardScanFrame:
    return CardScanFrame(nome=nome, mana=mana, descricao=None, tipo=tipo)


@pytest.fixture
def hand() -> list[CardScanFrame]:
    return [
        card("Tomo Vazio", 0, "tomo"),
        card("Espinafre", 1, "utilitario"),
        card("Otto", 2),
        card("Tomo Leve", 1, "tomo"),
    ]


def test_jogada_dentro_da_mana_passa(hand):
    assert validate(2, hand, 3) is None


def test_carta_cara_demais_e_rejeitada(hand):
    verdict = validate(2, hand, 1)
    assert isinstance(verdict, Rejection)
    assert "custa 2" in verdict.reason


def test_indice_fora_da_mao_e_rejeitado(hand):
    verdict = validate(9, hand, 5)
    assert isinstance(verdict, Rejection)
    assert "fora da mão" in verdict.reason


def test_indice_ausente_e_rejeitado(hand):
    assert isinstance(validate(None, hand, 5), Rejection)


def test_custo_desconhecido_nao_bloqueia(hand):
    hand.append(card("Ilegível", None))
    assert validate(len(hand) - 1, hand, 0) is None


def test_mana_desconhecida_nao_bloqueia(hand):
    assert validate(2, hand, None) is None


def test_affordable_filtra_por_mana(hand):
    assert affordable(hand, 1) == [0, 1, 3]
    assert affordable(hand, 0) == [0]


def test_fallback_prefere_tomo_mais_barato(hand):
    assert fallback_index(hand, 3) == 0


def test_fallback_usa_carta_mais_barata_sem_tomo():
    mao = [card("Caro", 5), card("Barato", 2)]
    assert fallback_index(mao, 3) == 1


def test_fallback_none_quando_nada_e_pagavel():
    assert fallback_index([card("Caro", 5)], 1) is None


def test_mana_implausivel_vira_none():
    """O modelo já devolveu mana=-1 lendo uma carta de bônus, onde o "-1" é o
    efeito ("custo reduzido em 1"), não o custo. Deixar passar faria `validate`
    aprovar a carta como se coubesse em qualquer mana."""
    assert CardScanFrame(nome="Bonus", mana=-1).mana is None
    assert CardScanFrame(nome="Absurda", mana=99).mana is None
    assert CardScanFrame(nome="Otto", mana=2).mana == 2
    assert CardScanFrame(nome="Tomo", mana=0).mana == 0


def test_carta_com_mana_invalida_nao_e_jogavel_por_engano():
    mao = [card("Bonus", -1)]
    assert mao[0].mana is None
    assert validate(0, mao, 0) is None, "mana desconhecida não bloqueia"
    assert affordable(mao, 0) == [0]


def test_regra_prefere_custo_conhecido_a_ilegivel():
    """Escolher uma carta de custo ilegível é apostar: se o custo real não couber,
    o jogo recusa a jogada em silêncio e o turno trava."""
    mao = [card("Ilegível", None), card("Otto", 2)]
    assert fallback_index(mao, 3) == 1


def test_tomo_conhecido_vence_ilegivel_mais_barato():
    mao = [card("Ilegível", None), card("Otto", 2), card("Tomo", 0, "tomo")]
    assert fallback_index(mao, 3) == 2


def test_ilegivel_e_usada_quando_nao_ha_alternativa():
    assert fallback_index([card("Ilegível", None)], 3) == 0


def test_prompt_de_combate_avisa_sobre_custo_desconhecido():
    from src.config import PATHS

    prompt = (PATHS.prompts / "combat_decide.txt").read_text(encoding="utf-8")
    assert "null" in prompt
    assert "custo conhecido" in prompt


@pytest.mark.parametrize(
    "grafia",
    ["jogar_carta", "jogarCarta", "JOGAR_CARTA", "jogar carta", "jogar-carta"],
)
def test_aceita_variacao_de_grafia_da_acao(grafia: str):
    """Observado no log: o modelo respondeu `jogarCarta`. A intenção estava certa
    e só a grafia errada, mas a validação recusava e queimava um ciclo inteiro de
    repergunta — que custa ~2s e conta como jogada ilegal na métrica."""
    from src.schemas import CombatAction

    assert CombatAction(acao=grafia, motivo="x").acao == "jogar_carta"


def test_acao_desconhecida_continua_recusada():
    """Normalizar grafia não pode virar aceitar qualquer coisa."""
    import pydantic

    from src.schemas import CombatAction

    with pytest.raises(pydantic.ValidationError):
        CombatAction(acao="descartar_mao", motivo="x")
