"""Posicionamento do cursor por identidade da carta."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import cv2
import pytest

from src import perception
from src.carddb import CardDB, CardRecord
from src.schemas import CardScanFrame
from src.vision.cards import card_bbox, detect_card_slots

FRAMES = Path(__file__).resolve().parent.parent / "frames"
# Sequência real: cursor em Gatti Amari, Giovanna, Phiera, Otto (direita→esquerda).
SEQUENCIA = [
    ("20260802T154032207_card_scan_1.png", "Gatti Amari"),
    ("20260802T154035945_card_scan_2.png", "Giovanna"),
    ("20260802T154039124_card_scan_3.png", "Phiera"),
    ("20260802T154042627_card_scan_4.png", "Otto"),
]
# Ordem esquerda→direita da mão, como o scan devolveria.
MAO = ["Magic", "Otto", "Phiera", "Giovanna", "Gatti Amari", "Pugnala"]


def caminho(name: str) -> Path:
    path = FRAMES / name
    if not path.is_file():
        pytest.skip(f"frame ausente: {name}")
    return path


@pytest.fixture
def db(tmp_path):
    """Cache preenchido com o recorte real de cada carta da sequência."""
    base = CardDB(tmp_path / "cards.json")
    for name, nome in SEQUENCIA:
        frame = cv2.imread(str(caminho(name)))
        circle = detect_card_slots(frame).selected
        x, y, w, h = card_bbox(circle)
        base.remember(
            frame[max(0, y) : y + h, max(0, x) : x + w],
            CardRecord(nome=nome, mana=1, descricao=None, tipo="ataque"),
        )
    return base


@pytest.fixture
def mao():
    return [
        CardScanFrame(nome=n, mana=1, descricao=None, tipo="ataque") for n in MAO
    ]


class Trilha:
    """Substitui `grab`: entrega os frames na ordem em que o cursor andaria."""

    def __init__(self, names: list[str]) -> None:
        self.fila = [caminho(n) for n in names]
        self.chamadas = 0

    def __call__(self, state: str | None = None) -> Path:
        self.chamadas += 1
        return self.fila.pop(0) if self.fila else self.fila_final

    @property
    def fila_final(self) -> Path:
        return caminho(SEQUENCIA[-1][0])


def test_para_de_imediato_se_ja_esta_na_carta(db, mao):
    """Nenhum input quando o cursor já está onde queremos."""
    trilha = Trilha([SEQUENCIA[0][0]])
    with (
        mock.patch.object(perception, "grab", trilha),
        mock.patch.object(perception, "gamepad") as pad,
    ):
        assert perception.seek_card(MAO.index("Gatti Amari"), mao, db)
    assert pad.tap_left.call_count == 0
    assert pad.tap_right.call_count == 0


def test_anda_pra_esquerda_ate_ver_a_carta(db, mao):
    """De Gatti Amari (idx 4) até Otto (idx 1): três passos à esquerda."""
    trilha = Trilha([s[0] for s in SEQUENCIA])
    with (
        mock.patch.object(perception, "grab", trilha),
        mock.patch.object(perception, "gamepad") as pad,
        mock.patch.object(perception, "_tap_and_wait", lambda x, left=True: (None, None)),
    ):
        pad.tap_left.side_effect = None
        assert perception.seek_card(MAO.index("Otto"), mao, db)
    assert trilha.chamadas == 4


def test_desiste_quando_a_carta_destacada_e_desconhecida(mao, tmp_path):
    """Sem referência no cache não dá pra decidir a direção — melhor refazer."""
    vazio = CardDB(tmp_path / "vazio.json")
    trilha = Trilha([SEQUENCIA[0][0]])
    with (
        mock.patch.object(perception, "grab", trilha),
        mock.patch.object(perception, "gamepad"),
    ):
        assert not perception.seek_card(0, mao, vazio)


def test_indice_fora_da_mao_falha_sem_tocar_no_controle(db, mao):
    with mock.patch.object(perception, "gamepad") as pad:
        assert not perception.seek_card(99, mao, db)
    assert pad.tap_left.call_count == 0


def test_sem_cache_nao_tenta_posicionar(mao):
    with mock.patch.object(perception, "gamepad") as pad, mock.patch.object(
        perception, "grab", Trilha([SEQUENCIA[0][0]])
    ):
        assert not perception.seek_card(0, mao, None)
    assert pad.tap_left.call_count == 0
