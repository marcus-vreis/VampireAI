"""Testes do consenso de contagem de cartas (k-vote)."""

from __future__ import annotations

from PIL import Image

from src.perception import _CountResult, _vote_count, enhance_for_count


def _r(total: int, idx: int | None = None) -> _CountResult:
    return _CountResult(total=total, indice_selecionada=idx)


def test_vote_majority_total() -> None:
    out = _vote_count([_r(4, 3), _r(4, 3), _r(5, 4)])
    assert out.total == 4
    assert out.indice_selecionada == 3


def test_vote_handles_idx_disagreement() -> None:
    out = _vote_count([_r(4, 3), _r(4, 2), _r(4, 3)])
    assert out.total == 4
    assert out.indice_selecionada == 3


def test_vote_all_null_idx() -> None:
    out = _vote_count([_r(3, None), _r(3, None)])
    assert out.total == 3
    assert out.indice_selecionada is None


def test_vote_empty_returns_zero() -> None:
    out = _vote_count([])
    assert out.total == 0
    assert out.indice_selecionada is None


def test_vote_single_sample() -> None:
    out = _vote_count([_r(2, 1)])
    assert out.total == 2
    assert out.indice_selecionada == 1


def test_enhance_returns_image_same_size() -> None:
    img = Image.new("RGB", (32, 16), (10, 20, 30))
    out = enhance_for_count(img)
    assert out.size == (32, 16)
