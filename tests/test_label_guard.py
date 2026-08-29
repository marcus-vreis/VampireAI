"""A rotulagem precisa capturar o JOGO, não o terminal."""

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
        assert label.capture_labeled("combat", None, None) is None
    assert not captura.exists(), "o frame ruim é apagado, não vai pro dataset"


def test_grava_quando_a_captura_e_o_jogo(captura):
    with (
        mock.patch.object(label, "signature", return_value=assinatura(Verdict.COMBAT)),
        mock.patch.object(label, "_observed", return_value={"cv_verdict": "combat"}),
    ):
        registro = label.capture_labeled("combat", 5, 2)
    assert registro is not None
    assert registro["state"] == "combat"
    assert registro["hand_size"] == 5


def test_a_sessao_nao_comeca_capturando_o_terminal(captura, capsys):
    """Melhor recusar de saída que descobrir depois de 50 frames rotulados."""
    with mock.patch.object(label, "signature", return_value=assinatura(Verdict.NOT_GAME)):
        assert label.session(ask_details=False) == 2
