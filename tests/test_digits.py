"""Leitura de algarismos do HUD por glifo aprendido."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from src.vision.digits import GlyphBook, find_glyphs, group_rows
from src.vision.hud import HEART_BOX, orb_glyphs, read_hp, text_mask

_RAIZ = Path(__file__).resolve().parent.parent
# Gabarito versionado. `frames/` é gitignored E rotacionado durante uma run,
# então teste que dependa dele passa a pular em silêncio — pior que falhar.
FRAMES = _RAIZ / "dataset" / "referencia"
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


DIM = "deck_baralho_referencia.png"  # HUD escurecido pelo painel aberto


def test_le_o_coracao_com_hud_escurecido():
    """O jogo escurece o HUD quando um painel abre.

    Um limiar absoluto de brilho perdia todos os quatro dígitos nessa condição —
    e era o que fazia o HP nunca ser lido.
    """
    from src.vision.digits import group_rows
    from src.vision.hud import heart_rows

    rows = heart_rows(load(DIM))
    assert len(rows) == 2
    assert [len(r) for r in rows] == [2, 2]
    assert group_rows(rows[0] + rows[1]) == rows


def test_orbe_sobrevive_ao_escurecimento():
    assert len(orb_glyphs(load(DIM))) == 1


def test_mesmo_digito_em_recortes_diferentes_e_reconhecido(book):
    """O coração mede 17px e é AMPLIADO até 12x18, o que gera ruído de
    quantização: o mesmo dígito não produz sempre o mesmo mapa de bits.
    A busca é por vizinho mais próximo, não por igualdade de chave."""
    normal = heart_glyphs(load())
    escuro = heart_glyphs(load(DIM))
    for glyph in normal:
        book.teach([glyph], 6 if glyph.x < 60 else 1)
        book.teach([glyph], 6 if glyph.x < 60 else 1)
    for glyph in escuro:
        esperado = 6 if glyph.x < 60 else 1
        assert book.lookup(glyph.key) == esperado


def test_digitos_diferentes_nao_se_confundem(book):
    """Piso medido: dígitos distintos ficam a 92 bits no mínimo, o limiar é 72."""
    seis = next(g for g in heart_glyphs(load()) if g.x < 60)
    um = next(g for g in heart_glyphs(load()) if g.x >= 60)
    book.teach([seis], 6)
    book.teach([seis], 6)
    assert book.lookup(um.key) is None


# Frame de combate real, capturado na primeira sessão de rotulagem. Coração
# mostra 61/61 — conferido olhando a imagem.
COMBATE = _RAIZ / "dataset" / "20260830T135106101_label_combat.png"


def test_le_o_coracao_em_combate_com_o_indicador_extra(book):
    """Em combate o jogo desenha mais um indicador dentro de `HEART_BOX`.

    Exigir "exatamente duas linhas" fazia `read_hp` devolver None justamente no
    único estado em que o HP importa pra decisão: o par de HP aparece em
    y=31..71, e o intruso em y=71..93, x=26 — uma terceira linha. Achado pela
    sessão de rotulagem, não por teste sintético.
    """
    if not COMBATE.is_file():
        pytest.skip("frame de combate ausente")
    from src.vision.hud import heart_rows

    frame = cv2.imread(str(COMBATE))
    rows = heart_rows(frame)
    assert len(rows) == 3, "o indicador extra continua entrando na caixa do coração"
    for row in rows[:2]:
        book.teach(row, 61)
        book.teach(row, 61)
    assert read_hp(frame, book) == (61, 61)
