"""Cache de identidade de carta por hash perceptual."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.carddb import CardDB, CardRecord, perceptual_hash
from src.vision.cards import card_bbox, detect_card_slots

FRAMES = Path(__file__).resolve().parent.parent / "frames"


def art(seed: int, w: int = 220, h: int = 310) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


@pytest.fixture
def db(tmp_path):
    return CardDB(tmp_path / "cards.json")


def test_carta_desconhecida_da_miss(db):
    assert db.lookup(art(1)) is None


def test_carta_lembrada_e_encontrada(db):
    card = art(2)
    db.remember(card, CardRecord(nome="Otto", mana=2, descricao="374 de dano", tipo="ataque"))
    hit = db.lookup(card)
    assert hit is not None
    assert hit.nome == "Otto"
    assert hit.mana == 2


def test_cartas_diferentes_nao_colidem(db):
    db.remember(art(3), CardRecord(nome="A", mana=0, descricao=None, tipo="tomo"))
    assert db.lookup(art(4)) is None


def test_tolera_deslocamento_do_recorte(db):
    """Recorte real: o detector varia um ou dois pixels entre frames.

    Precisa ser arte de verdade, não ruído — ruído aleatório não tem estrutura
    espacial, então qualquer deslocamento troca o hash inteiro e o teste mediria
    o fixture em vez do cache.
    """
    frame = FRAMES / "20260802T154032207_card_scan_1.png"
    if not frame.is_file():
        pytest.skip("frame de referência ausente")
    img = cv2.imread(str(frame))
    slots = detect_card_slots(img)
    x, y, w, h = card_bbox(slots.selected)
    db.remember(img[y : y + h, x : x + w], CardRecord(nome="Gatti Amari", mana=1, descricao=None, tipo="ataque"))
    deslocado = img[y + 2 : y + h + 2, x + 1 : x + w + 1]
    hit = db.lookup(deslocado)
    assert hit is not None and hit.nome == "Gatti Amari"


def test_carta_diferente_do_mesmo_frame_nao_colide(db):
    frame = FRAMES / "20260802T154032207_card_scan_1.png"
    if not frame.is_file():
        pytest.skip("frame de referência ausente")
    img = cv2.imread(str(frame))
    circles = detect_card_slots(img).circles
    a, b = card_bbox(circles[0]), card_bbox(circles[-1])
    db.remember(img[a[1] : a[1] + a[3], a[0] : a[0] + a[2]],
                CardRecord(nome="primeira", mana=0, descricao=None, tipo="tomo"))
    assert db.lookup(img[b[1] : b[1] + b[3], b[0] : b[0] + b[2]]) is None


def test_persiste_entre_instancias(tmp_path):
    path = tmp_path / "cards.json"
    card = art(6)
    CardDB(path).remember(card, CardRecord(nome="Vela", mana=0, descricao=None, tipo="bonus"))
    recarregada = CardDB(path)
    assert len(recarregada) == 1
    assert recarregada.lookup(card).nome == "Vela"


def test_hash_e_estavel():
    card = art(7)
    assert perceptual_hash(card) == perceptual_hash(card.copy())
