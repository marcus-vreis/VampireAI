"""Espera adaptativa pelo movimento do cursor durante a travessia da mão."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import cv2
import pytest

from src import perception
from src.vision.cards import CostCircle

FRAMES = Path(__file__).resolve().parent.parent / "frames"
# Sequência real de scan: o cursor anda uma carta pra esquerda a cada passo.
SEQUENCIA = [
    "20260802T154032207_card_scan_1.png",
    "20260802T154035945_card_scan_2.png",
    "20260802T154039124_card_scan_3.png",
    "20260802T154042627_card_scan_4.png",
]


def caminho(name: str) -> Path:
    path = FRAMES / name
    if not path.is_file():
        pytest.skip(f"frame de referência ausente: {name}")
    return path


class FonteDeFrames:
    """Substitui `grab`: devolve os frames da sequência, um por chamada."""

    def __init__(self, names: list[str], repeticoes: int = 1) -> None:
        self.fila = [caminho(n) for n in names for _ in range(repeticoes)]
        self.chamadas = 0

    def __call__(self, state: str | None = None) -> Path:
        self.chamadas += 1
        return self.fila.pop(0) if self.fila else caminho(SEQUENCIA[-1])


def test_para_assim_que_o_cursor_anda():
    """Uma captura basta quando o jogo já respondeu — não espera tempo fixo."""
    fonte = FonteDeFrames([SEQUENCIA[1]])
    anterior = cv2.imread(str(caminho(SEQUENCIA[0])))
    x_anterior = perception.detect_card_slots(anterior).selected.x

    with mock.patch.object(perception, "grab", fonte), mock.patch.object(perception, "gamepad"):
        _frame, selecionada = perception._tap_left_and_wait(x_anterior)

    assert fonte.chamadas == 1
    assert selecionada is not None
    assert abs(selecionada.x - x_anterior) >= perception._SAME_CARD_PX


def test_insiste_enquanto_o_cursor_nao_saiu_do_lugar():
    """O sleep fixo lia um frame ainda em animação; aqui a captura se repete."""
    fonte = FonteDeFrames([SEQUENCIA[0], SEQUENCIA[0], SEQUENCIA[1]])
    anterior = cv2.imread(str(caminho(SEQUENCIA[0])))
    x_anterior = perception.detect_card_slots(anterior).selected.x

    with mock.patch.object(perception, "grab", fonte), mock.patch.object(perception, "gamepad"):
        _frame, selecionada = perception._tap_left_and_wait(x_anterior)

    assert fonte.chamadas == 3
    assert abs(selecionada.x - x_anterior) >= perception._SAME_CARD_PX


def test_desiste_no_teto_quando_o_cursor_nunca_anda():
    """Na ponta do leque o cursor não se move; a travessia tem que terminar."""
    fonte = FonteDeFrames([SEQUENCIA[0]], repeticoes=200)
    anterior = cv2.imread(str(caminho(SEQUENCIA[0])))
    x_anterior = perception.detect_card_slots(anterior).selected.x

    with (
        mock.patch.object(perception, "grab", fonte),
        mock.patch.object(perception, "gamepad"),
        mock.patch.object(perception, "_CURSOR_MOVE_TIMEOUT_S", 0.15),
    ):
        _frame, selecionada = perception._tap_left_and_wait(x_anterior)

    assert selecionada is not None
    assert abs(selecionada.x - x_anterior) < perception._SAME_CARD_PX


def test_travessia_completa_percorre_a_mao_uma_vez():
    """A travessia para sozinha ao voltar a uma carta já vista — sem saber o total."""
    fonte = FonteDeFrames(SEQUENCIA + [SEQUENCIA[-1]])
    lido = CostCircle(x=0, y=0, w=20, h=20)
    with (
        mock.patch.object(perception, "grab", fonte),
        mock.patch.object(perception, "gamepad"),
        mock.patch.object(perception, "read_card") as ler,
        mock.patch.object(perception, "read_mana_hybrid", return_value=3),
        mock.patch.object(perception, "read_hp_hybrid", return_value=(61, 61)),
    ):
        ler.side_effect = lambda *a, **k: perception.CardScanFrame(
            nome=f"carta{ler.call_count}", mana=1, descricao=None, tipo="ataque"
        )
        scan = perception.scan_combat_hand()

    assert len(scan.cards) == len(SEQUENCIA)
    assert scan.mana == 3
    assert scan.hp == (61, 61)
    assert lido is not None
