"""Percepção determinística por visão computacional.

O VLM é caro (~2.8s/chamada) e erra em geometria: contar cartas, achar o cursor,
ler o minimapa. Tudo isso é pixel art de assinatura estável — código resolve em
milissegundos e sem alucinar. O VLM fica só com semântica.
"""

from src.vision.cards import CardSlots, CostCircle, card_bbox, detect_card_slots
from src.vision.screen import ScreenSignature, Verdict, signature

__all__ = [
    "CardSlots",
    "CostCircle",
    "ScreenSignature",
    "Verdict",
    "card_bbox",
    "detect_card_slots",
    "signature",
]
