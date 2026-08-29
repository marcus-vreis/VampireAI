"""Regras de combate: validação da jogada escolhida e plano de reserva.

O VLM continua decidindo. O código só recusa o que o jogo recusaria — mana
insuficiente, índice fora da mão — e, se o modelo insistir no inválido, joga pela
regra documentada em `jogo.md`: custo crescente, tomos primeiro.

Manter a decisão com o modelo (em vez de calcular a jogada ótima) preserva o
ângulo de pesquisa do projeto; a taxa de rejeição vira, de graça, uma medida de
quão bom o modelo é na tarefa.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.schemas import CardScanFrame


@dataclass(frozen=True)
class Rejection:
    reason: str


def validate(
    index: int | None, hand: list[CardScanFrame], mana: int | None
) -> Rejection | None:
    """None se a jogada é legal; Rejection com o motivo em PT-BR se não é."""
    if index is None:
        return Rejection("nenhum índice de carta foi informado")
    if not 0 <= index < len(hand):
        return Rejection(
            f"índice {index} fora da mão (a mão tem {len(hand)} cartas, "
            f"índices 0 a {len(hand) - 1})"
        )
    cost = hand[index].mana
    if cost is not None and mana is not None and cost > mana:
        return Rejection(
            f"a carta '{hand[index].nome}' custa {cost} de mana e você só tem {mana}"
        )
    return None


def affordable(hand: list[CardScanFrame], mana: int | None) -> list[int]:
    """Índices que a mana atual permite jogar."""
    if mana is None:
        return list(range(len(hand)))
    return [i for i, c in enumerate(hand) if c.mana is None or c.mana <= mana]


def fallback_index(hand: list[CardScanFrame], mana: int | None) -> int | None:
    """Jogada pela regra do jogo quando o modelo não produz uma legal.

    `jogo.md`: jogar em ordem CRESCENTE de custo buffa as cartas seguintes, e
    tomos (vermelhos, custo baixo) devolvem mana. Então: tomo mais barato
    primeiro; sem tomo jogável, a carta jogável mais barata.

    Carta de custo ILEGÍVEL fica por último. `validate` não a bloqueia — não dá
    pra provar que é ilegal — mas escolher uma de propósito é apostar: se o custo
    real não couber, o jogo recusa a jogada em silêncio e o turno trava. Entre
    uma carta que sabemos jogável e uma que não sabemos, a certa é a conhecida.
    """
    playable = affordable(hand, mana)
    if not playable:
        return None
    known = [i for i in playable if hand[i].mana is not None]
    pool = known or playable
    tomes = [i for i in pool if hand[i].tipo == "tomo"]
    pool = tomes or pool
    return min(pool, key=lambda i: (hand[i].mana if hand[i].mana is not None else 99, i))
