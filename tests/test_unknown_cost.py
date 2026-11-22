"""Custo ilegível é último recurso, não carta proibida (ADR-051, ADR-105)."""

from __future__ import annotations

from unittest import mock

import pytest

from src import agent
from src.perception import HandScan
from src.schemas import CardScanFrame, CombatAction


def carta(nome: str, mana: int | None) -> CardScanFrame:
    return CardScanFrame(nome=nome, mana=mana, descricao=None, tipo="ataque")


@pytest.fixture(autouse=True)
def limpa():
    agent.forget_hand()
    yield
    agent.forget_hand()


def jogar(mao: HandScan, acao: CombatAction):
    """Roda handle_combat com a decisão do modelo fixada e o controle falso."""
    with (
        mock.patch.object(agent, "_current_hand", return_value=mao),
        mock.patch.object(agent, "_decide_combat", return_value=acao),
        mock.patch.object(agent, "seek_card", return_value=True) as seek,
        mock.patch.object(agent, "input_exec") as controle,
        mock.patch.object(agent, "default_carddb"),
    ):
        agent.handle_combat(None)
    return seek, controle


FIM = CombatAction(acao="finalizar_turno", indice_alvo=None, motivo="mana insuficiente")


def test_modelo_encerra_mas_ha_carta_de_custo_ilegivel_tenta_ela():
    """Run de 2026-11-22: "Finalizando turno: mana insuficiente e cartas com custo
    desconhecido", com 4 de mana sobrando. Encerrar ali desperdiça o turno."""
    mao = HandScan(cards=[carta("Otto", 5), carta("Magicatalisador", None)], cursor_idx=1, mana=4)
    seek, controle = jogar(mao, FIM)
    assert seek.call_args.args[0] == 1, "tenta a carta de custo ilegível"
    assert controle.confirm.call_count == 1
    assert controle.end_turn.call_count == 0


def test_sem_carta_de_custo_ilegivel_encerra_normalmente():
    mao = HandScan(cards=[carta("Otto", 5), carta("Faca", 6)], cursor_idx=1, mana=4)
    seek, controle = jogar(mao, FIM)
    assert seek.call_count == 0
    assert controle.end_turn.call_count == 1


def test_carta_tentada_sai_da_mao_conhecida_e_nao_repete():
    """Se o jogo recusar, a carta continua na mão real; tirá-la da conhecida é o
    que impede tentar de novo no mesmo turno. A próxima travessia a traz de volta."""
    mao = HandScan(cards=[carta("Otto", 5), carta("Magicatalisador", None)], cursor_idx=1, mana=4)
    jogar(mao, FIM)
    assert [c.nome for c in agent._HAND.cards] == ["Otto"]
    seek, controle = jogar(agent._HAND, FIM)
    assert seek.call_count == 0
    assert controle.end_turn.call_count == 1
