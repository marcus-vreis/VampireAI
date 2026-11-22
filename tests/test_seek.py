"""Posicionamento do cursor por identidade da carta."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import cv2
import pytest

from src import perception
from src.carddb import CardDB, CardRecord, perceptual_hash
from src.schemas import CardScanFrame
from src.vision.cards import card_bbox, detect_card_slots

_RAIZ = Path(__file__).resolve().parent.parent
# Gabarito versionado. `frames/` é gitignored E rotacionado durante uma run,
# então teste que dependa dele passa a pular em silêncio — pior que falhar.
FRAMES = _RAIZ / "dataset" / "referencia"
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
    return [CardScanFrame(nome=n, mana=1, descricao=None, tipo="ataque") for n in MAO]


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


class ModeloFalso:
    """Substitui `ask_vlm` na leitura de carta: responde um nome fixo e conta."""

    def __init__(self, nome: str | None) -> None:
        self.nome = nome
        self.chamadas = 0

    def __call__(self, *args, **kwargs) -> dict:
        self.chamadas += 1
        if self.nome is None:
            raise RuntimeError("VLM falhou 3x")
        return {"nome": self.nome, "mana": 1, "descricao": None, "tipo": "ataque"}


def test_carta_fora_do_cache_e_lida_pelo_modelo_e_aprendida(mao, tmp_path):
    """O mesmo sprite chega a 88 bits de distância conforme o contexto (run de
    2026-11-22), e desistir ali custava travessia + nova decisão. Ler custa uma
    chamada, e ensina o cache pra próxima vez."""
    vazio = CardDB(tmp_path / "vazio.json")
    modelo = ModeloFalso("Gatti Amari")
    with (
        mock.patch.object(perception, "grab", Trilha([SEQUENCIA[0][0]])),
        mock.patch.object(perception, "gamepad"),
        mock.patch.object(perception, "ask_vlm", modelo),
    ):
        assert perception.seek_card(MAO.index("Gatti Amari"), mao, vazio)
    assert modelo.chamadas == 1
    assert len(vazio) == 1, "a variante nova do sprite tem que ficar no cache"


def test_desiste_quando_nem_o_modelo_identifica_a_carta(mao, tmp_path):
    vazio = CardDB(tmp_path / "vazio.json")
    with (
        mock.patch.object(perception, "grab", Trilha([SEQUENCIA[0][0]])),
        mock.patch.object(perception, "gamepad"),
        mock.patch.object(perception, "ask_vlm", ModeloFalso(None)),
    ):
        assert not perception.seek_card(0, mao, vazio)


def test_carta_comprada_depois_da_jogada_nao_interrompe_a_busca(db):
    """Giovanna e Phiera entraram na mão depois da última travessia, então não
    estão na mão conhecida. A busca segue na mesma direção até achar Otto."""
    mao_velha = [
        CardScanFrame(nome=n, mana=1, descricao=None, tipo="ataque")
        for n in ["Magic", "Otto", "Gatti Amari", "Pugnala"]
    ]
    trilha = Trilha([s[0] for s in SEQUENCIA])
    with (
        mock.patch.object(perception, "grab", trilha),
        mock.patch.object(perception, "gamepad"),
        mock.patch.object(perception, "_tap_and_wait", lambda x, left=True: (None, None)),
    ):
        assert perception.seek_card(1, mao_velha, db)
    assert trilha.chamadas == 4, "Gatti -> Giovanna -> Phiera -> Otto"


def test_indice_fora_da_mao_falha_sem_tocar_no_controle(db, mao):
    with mock.patch.object(perception, "gamepad") as pad:
        assert not perception.seek_card(99, mao, db)
    assert pad.tap_left.call_count == 0


def test_sem_cache_nao_tenta_posicionar(mao):
    with (
        mock.patch.object(perception, "gamepad") as pad,
        mock.patch.object(perception, "grab", Trilha([SEQUENCIA[0][0]])),
    ):
        assert not perception.seek_card(0, mao, None)
    assert pad.tap_left.call_count == 0


def test_entra_na_mao_se_o_cursor_comeca_fora(db, mao):
    """Depois de jogar uma carta o cursor pode cair fora do leque; `seek_card`
    precisa pôr o cursor de volta na mão antes de desistir."""
    sem_cursor = _RAIZ / "dataset" / "20260830T135106101_label_combat.png"
    com_cursor = caminho(SEQUENCIA[0][0])
    if not sem_cursor.is_file():
        pytest.skip("frame sem cursor ausente")
    img_com = cv2.imread(str(com_cursor))
    selecionada = detect_card_slots(img_com).selected
    with (
        mock.patch.object(perception, "grab", return_value=sem_cursor),
        mock.patch.object(perception, "_enter_hand", return_value=(img_com, selecionada)) as entrar,
    ):
        assert perception.seek_card(MAO.index("Gatti Amari"), mao, db)
    assert entrar.call_count == 1


def test_tolera_pequena_variacao_de_grafia_no_nome(db):
    """O VLM às vezes transcreve 'Torno Leve' num turno e 'Tomo Leve' no outro."""
    mao_typo = [
        CardScanFrame(nome="Gati Amari", mana=1, descricao=None, tipo="ataque"),
    ]
    trilha = Trilha([SEQUENCIA[0][0]])
    with (
        mock.patch.object(perception, "grab", trilha),
        mock.patch.object(perception, "gamepad"),
    ):
        assert perception.seek_card(0, mao_typo, db)


def _hash_do_frame(name: str) -> str:
    frame = cv2.imread(str(caminho(name)))
    return perceptual_hash(perception._crop_card(frame, detect_card_slots(frame).selected))


def test_hash_da_mao_localiza_carta_com_nome_mal_lido(tmp_path):
    """Run de 2026-11-22: o VLM leu Otto como '462 de dano.' (a descrição), e a
    busca, sem achar o nome na mão, andava às cegas pro lado errado. O hash da
    própria travessia diz onde o cursor está sem depender do texto."""
    mal_lido = CardDB(tmp_path / "cards.json")
    otto = cv2.imread(str(caminho(SEQUENCIA[3][0])))
    mal_lido.remember(
        perception._crop_card(otto, detect_card_slots(otto).selected),
        CardRecord(nome="462 de dano.", mana=2, descricao=None, tipo="ataque"),
    )
    mao = [CardScanFrame(nome=n, mana=1, descricao=None, tipo="ataque") for n in MAO]
    longe = "00" * 32
    hashes = [longe, _hash_do_frame(SEQUENCIA[3][0]), longe, longe, longe, longe]
    hashes[MAO.index("Gatti Amari")] = _hash_do_frame(SEQUENCIA[0][0])
    trilha = Trilha([SEQUENCIA[3][0], SEQUENCIA[0][0]])
    lados = []
    with (
        mock.patch.object(perception, "grab", trilha),
        mock.patch.object(perception, "gamepad"),
        mock.patch.object(perception, "ask_vlm", ModeloFalso(None)),
        mock.patch.object(
            perception, "_tap_and_wait", lambda x, left=True: lados.append(left) or (None, None)
        ),
    ):
        assert perception.seek_card(MAO.index("Gatti Amari"), mao, mal_lido, hashes=hashes)
    assert lados == [False], "de Otto (1) até Gatti (4) é pra direita"


def test_desempata_cartas_repetidas_pela_posicao_no_leque():
    """Com duas cópias da mesma carta nas pontas (índices 0 e 4) e o cursor na
    direita, o índice atual tem que ser 4 (para andar à esquerda rumo ao alvo 2)."""
    from src.vision.cards import CardSlots, CostCircle

    mao_dupla = [
        CardScanFrame(nome=n, mana=1, descricao=None, tipo="ataque")
        for n in ["Alho", "Tomo", "Faca", "Otto", "Alho"]
    ]
    circles = [CostCircle(x=300 + 100 * i, y=500, w=30 if i == 4 else 22, h=22) for i in range(5)]
    slots = CardSlots(circles=circles, selected_idx=4)
    assert perception._current_hand_index("Alho", mao_dupla, slots) == 4
