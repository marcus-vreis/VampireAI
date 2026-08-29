"""Leitura de algarismos do HUD por glifo aprendido."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from src.vision.digits import GlyphBook, find_glyphs, group_rows
from src.vision.hud import HEART_BOX, orb_glyphs, read_hp, text_mask

FRAMES = Path(__file__).resolve().parent.parent / "frames"
# Frame conferido olhando a imagem: orbe mostra 4, coração mostra 61/61.
REFERENCE = "20260802T154240385_combat_initial.png"


def load(name: str = REFERENCE):
    path = FRAMES / name
    if not path.is_file():
        pytest.skip(f"frame de referência ausente: {name}")
    return cv2.imread(str(path))


@pytest.fixture
def book(tmp_path):
    return GlyphBook(tmp_path / "glyphs.json")


def heart_glyphs(frame):
    x0, y0, x1, y1 = HEART_BOX
    return find_glyphs(text_mask(frame[y0:y1, x0:x1]))


def test_orbe_tem_um_algarismo():
    assert len(orb_glyphs(load())) == 1


def test_coracao_tem_quatro_algarismos_em_duas_linhas():
    """O contorno branco do coração cai na mesma máscara e precisa ser filtrado."""
    rows = group_rows(heart_glyphs(load()))
    assert len(rows) == 2
    assert [len(r) for r in rows] == [2, 2]


def test_glifo_desconhecido_nao_e_lido(book):
    assert book.read(orb_glyphs(load())) is None


def test_um_voto_nao_basta(book):
    """Uma leitura errada do modelo envenenaria o glifo — exige confirmação."""
    glyphs = orb_glyphs(load())
    book.teach(glyphs, 4)
    assert book.read(glyphs) is None
    book.teach(glyphs, 4)
    assert book.read(glyphs) == 4


def test_voto_majoritario_vence(book):
    glyphs = orb_glyphs(load())
    book.teach(glyphs, 4)
    book.teach(glyphs, 9)  # leitura errada isolada
    book.teach(glyphs, 4)
    assert book.read(glyphs) == 4


def test_teach_ignora_quando_contagem_diverge(book):
    """Se a segmentação vê 1 glifo e a leitura diz '42', não dá pra atribuir."""
    glyphs = orb_glyphs(load())
    book.teach(glyphs, 42)
    book.teach(glyphs, 42)
    assert book.read(glyphs) is None


def test_le_hp_de_duas_linhas(book):
    frame = load()
    rows = group_rows(heart_glyphs(frame))
    for row in rows:
        book.teach(row, 61)
        book.teach(row, 61)
    assert read_hp(frame, book) == (61, 61)


def test_hp_sem_livro_e_none():
    assert read_hp(load()) is None


def test_livro_persiste_entre_instancias(tmp_path):
    path = tmp_path / "glyphs.json"
    glyphs = orb_glyphs(load())
    first = GlyphBook(path)
    first.teach(glyphs, 4)
    first.teach(glyphs, 4)
    assert GlyphBook(path).read(glyphs) == 4
