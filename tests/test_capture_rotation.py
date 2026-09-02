"""Rotação dos frames de debug."""

from __future__ import annotations

from unittest import mock

import pytest

from src import capture


@pytest.fixture
def pasta(tmp_path):
    """`PATHS` é dataclass congelada, então trocamos a referência do módulo."""
    from types import SimpleNamespace

    with mock.patch.object(capture, "PATHS", SimpleNamespace(frames=tmp_path)):
        capture._since_prune = 0
        yield tmp_path


def criar(pasta, n: int, mtime_base: float = 1000.0) -> None:
    import os

    for i in range(n):
        f = pasta / f"frame_{i:04d}.png"
        f.write_bytes(b"x")
        os.utime(f, (mtime_base + i, mtime_base + i))


def test_apaga_os_mais_antigos_acima_do_limite(pasta):
    """Uma run de 1h escreveria ~2.7 GB sem isto — contra os 10 GB que o README
    pede de disco no total."""
    criar(pasta, 60)
    with (
        mock.patch.object(capture, "_KEEP_FRAMES", 20),
        mock.patch.object(capture, "_PRUNE_EVERY", 1),
    ):
        capture._prune()
    restantes = sorted(p.name for p in pasta.glob("*.png"))
    assert len(restantes) == 20
    assert restantes[0] == "frame_0040.png", "sobram os mais recentes"


def test_nao_lista_o_diretorio_a_cada_captura(pasta):
    """Listar milhares de arquivos a cada captura custaria mais que a captura."""
    criar(pasta, 60)
    with (
        mock.patch.object(capture, "_KEEP_FRAMES", 20),
        mock.patch.object(capture, "_PRUNE_EVERY", 50),
    ):
        for _ in range(10):
            capture._prune()
    assert len(list(pasta.glob("*.png"))) == 60, "só poda a cada _PRUNE_EVERY"


def test_limite_zero_desliga_a_rotacao(pasta):
    criar(pasta, 30)
    with (
        mock.patch.object(capture, "_KEEP_FRAMES", 0),
        mock.patch.object(capture, "_PRUNE_EVERY", 1),
    ):
        capture._prune()
    assert len(list(pasta.glob("*.png"))) == 30


def test_arquivo_travado_nao_derruba_a_run(pasta):
    """Windows trava arquivo aberto; uma run não pode morrer por causa disso."""
    criar(pasta, 30)
    with (
        mock.patch.object(capture, "_KEEP_FRAMES", 5),
        mock.patch.object(capture, "_PRUNE_EVERY", 1),
        mock.patch.object(capture.Path, "unlink", side_effect=OSError("em uso")),
    ):
        capture._prune()  # não levanta
