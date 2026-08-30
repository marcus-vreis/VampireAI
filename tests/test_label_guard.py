"""A rotulagem precisa capturar o JOGO, não o terminal — e gravar o que a pessoa
realmente respondeu, sem confundir "nenhuma" com "não sei"."""

from __future__ import annotations

from unittest import mock

import pytest

from src import label
from src.vision.screen import Verdict


def assinatura(verdict):
    return mock.Mock(verdict=verdict)


@pytest.fixture
def captura(tmp_path):
    shot = tmp_path / "x.png"
    shot.write_bytes(b"x")
    with (
        mock.patch.object(label, "grab", return_value=shot),
        mock.patch.object(label, "cv2"),
        mock.patch.object(label, "DATASET_DIR", tmp_path / "ds"),
        mock.patch.object(label, "LABELS_FILE", tmp_path / "ds" / "labels.jsonl"),
    ):
        yield shot


def test_recusa_gravar_quando_a_captura_nao_e_o_jogo(captura):
    """Pra ler a tecla o terminal precisa do foco, então o jogo nunca está
    focado durante a rotulagem. Sem esta checagem, um jogo atrás do terminal
    produziria um dataset inteiro de prints do terminal rotulados como combate."""
    with mock.patch.object(label, "signature", return_value=assinatura(Verdict.NOT_GAME)):
        assert label.capture_labeled("combat") is None
    assert not captura.exists(), "o frame ruim é apagado, não vai pro dataset"


def test_grava_quando_a_captura_e_o_jogo(captura):
    with (
        mock.patch.object(label, "signature", return_value=assinatura(Verdict.COMBAT)),
        mock.patch.object(label, "_observed", return_value={"cv_verdict": "combat"}),
    ):
        alvo = label.capture_labeled("combat")
        assert alvo is not None
        registro = label.registrar(alvo, "combat", {"hand_size": 5, "hand_size_known": True})
    assert registro["state"] == "combat"
    assert registro["hand_size"] == 5


def test_a_sessao_nao_comeca_capturando_o_terminal(captura, capsys):
    """Melhor recusar de saída que descobrir depois de 50 frames rotulados."""
    with mock.patch.object(label, "signature", return_value=assinatura(Verdict.NOT_GAME)):
        assert label.session(ask_details=False) == 2


def test_nenhuma_selecionada_e_resposta_nao_falta():
    """O caso que confundiu na primeira sessão: num combate com o cursor em
    "Finalizar turno" nenhuma carta está levantada. Isso é gabarito válido, e
    tratar como "não respondi" apagaria o caso mais comum de cursor nulo."""
    with mock.patch.object(label, "input", return_value="n", create=True):
        valor, respondeu = label._perguntar_selecionada("")
    assert valor is None and respondeu is True

    with mock.patch.object(label, "input", return_value="", create=True):
        valor, respondeu = label._perguntar_selecionada("")
    assert valor is None and respondeu is False


def test_a_pergunta_e_1_based_e_o_dado_e_0_based():
    """Contar a partir de 1 é natural pra quem olha a tela; o resto do código
    indexa a partir de 0. A conversão fica aqui, num lugar só."""
    with mock.patch.object(label, "input", return_value="1", create=True):
        assert label._perguntar_selecionada("") == (0, True)
    with mock.patch.object(label, "input", return_value="4", create=True):
        assert label._perguntar_selecionada("") == (3, True)


def test_rotulo_antigo_sem_flag_ainda_conta_como_resposta():
    """A primeira sessão gravou antes de existir o par `_known`. Ali, valor
    presente já significava resposta — o resumo não pode descartar esses frames."""
    antigo = {"hand_size": 6, "cursor": None}
    assert label._sabe(antigo, "hand_size") is True
    assert label._sabe(antigo, "cursor") is False

    novo = {"cursor": None, "cursor_known": True}
    assert label._sabe(novo, "cursor") is True


def test_a_acuracia_ignora_frames_sem_gabarito():
    """Um frame sem resposta não é acerto nem erro. Contar como erro puniria a
    CV por uma pergunta que ninguém respondeu."""
    records = [
        {"cursor": 2, "cursor_known": True, "cv_cursor": 2},
        {"cursor": None, "cursor_known": False, "cv_cursor": 9},
    ]
    assert "1/1" in label._acuracia(records, "cursor", "cv_cursor")


def test_a_cobertura_diz_o_que_falta():
    """A pergunta "é pra fazer o jogo todo?" precisa ter resposta na tela."""
    feito = dict.fromkeys(label._META, 99)
    assert "completa" in label._tabela_cobertura(feito)
    assert "combat" in label._tabela_cobertura({"combat": 0})
