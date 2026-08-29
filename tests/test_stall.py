"""Detecção de travamento e escalonamento de botões."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.stall import Nudge, StallDetector

_RAIZ = Path(__file__).resolve().parent.parent
# Gabarito versionado. `frames/` é gitignored E rotacionado durante uma run,
# então teste que dependa dele passa a pular em silêncio — pior que falhar.
FRAMES = _RAIZ / "dataset" / "referencia"

# Pares medidos em frames reais. Delta da assinatura 16x16:
# mesma tela (só animação de fundo) 0.89-1.46; cursor andando uma carta 4.44-8.43.
PARADO = ("20260802T153918887_card_scan_1.png", "20260802T153921862_card_scan_2.png")
MUDOU = ("20260802T154032207_card_scan_1.png", "20260802T154035945_card_scan_2.png")


def load(name: str) -> np.ndarray:
    path = FRAMES / name
    if not path.is_file():
        pytest.skip(f"frame de referência ausente: {name}")
    return cv2.imread(str(path))


def test_frame_repetido_marca_travamento():
    frame = load(PARADO[0])
    d = StallDetector(patience=2)
    for _ in range(3):
        d.observe(frame)
    assert d.stuck


def test_animacao_de_fundo_conta_como_mesma_tela():
    """Dois frames da mesma tela parada diferem só por partícula — é travamento."""
    d = StallDetector(patience=2)
    d.observe(load(PARADO[0]))
    d.observe(load(PARADO[1]))
    d.observe(load(PARADO[0]))
    assert d.stuck


def test_cursor_andando_uma_carta_nao_e_travamento():
    """A menor mudança real que o detector precisa enxergar."""
    d = StallDetector(patience=2)
    d.observe(load(MUDOU[0]))
    d.observe(load(MUDOU[1]))
    assert not d.stuck


def test_mudanca_zera_a_contagem():
    d = StallDetector(patience=2)
    for _ in range(3):
        d.observe(load(PARADO[0]))
    assert d.stuck
    d.observe(load(MUDOU[1]))
    assert not d.stuck


def test_escalona_botoes_em_ordem():
    frame = load(PARADO[0])
    d = StallDetector(patience=1)
    d.observe(frame)
    d.observe(frame)
    assert d.next_nudge() is Nudge.CONFIRM
    assert d.next_nudge() is Nudge.CANCEL
    assert d.next_nudge() is Nudge.FORWARD
    assert d.next_nudge() is None
    assert d.exhausted


def test_nao_empurra_quando_nao_esta_travado():
    d = StallDetector(patience=2)
    d.observe(load(PARADO[0]))
    assert d.next_nudge() is None
    assert not d.exhausted


def test_reset_limpa_estado():
    frame = load(PARADO[0])
    d = StallDetector(patience=1)
    d.observe(frame)
    d.observe(frame)
    assert d.stuck
    d.reset()
    assert not d.stuck
    assert d.next_nudge() is None
