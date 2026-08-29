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
