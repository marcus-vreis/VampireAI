"""Testes do navigate_horizontal e select_and_confirm.

Monkeypatcha as primitivas do gamepad para não tocar em vgamepad/ViGEm.
"""

from __future__ import annotations

import pytest

from src import gamepad, input_exec


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int | None]]:
    log: list[tuple[str, int | None]] = []

    def fake_tap_left(times: int = 1) -> None:
        log.append(("tap_left", times))

    def fake_tap_right(times: int = 1) -> None:
        log.append(("tap_right", times))

    def fake_confirm() -> None:
        log.append(("confirm", None))

    monkeypatch.setattr(gamepad, "tap_left", fake_tap_left)
    monkeypatch.setattr(gamepad, "tap_right", fake_tap_right)
    monkeypatch.setattr(gamepad, "confirm", fake_confirm)

    monkeypatch.setattr(input_exec.time, "sleep", lambda *_: None)
    return log


def test_navigate_zero_is_noop(calls: list[tuple[str, int | None]]) -> None:
    input_exec.navigate_horizontal(0)
    assert calls == []


def test_navigate_positive_taps_right(calls: list[tuple[str, int | None]]) -> None:
    input_exec.navigate_horizontal(3)
    assert calls == [("tap_right", 3)]


def test_navigate_negative_taps_left(calls: list[tuple[str, int | None]]) -> None:
    input_exec.navigate_horizontal(-2)
    assert calls == [("tap_left", 2)]


def test_select_and_confirm_zero_just_presses_x(
    calls: list[tuple[str, int | None]],
) -> None:
    input_exec.select_and_confirm(0)
    assert calls == [("confirm", None)]


def test_select_and_confirm_moves_then_x(
    calls: list[tuple[str, int | None]],
) -> None:
    input_exec.select_and_confirm(2)
    assert calls == [("tap_right", 2), ("confirm", None)]


def test_select_and_confirm_negative_then_x(
    calls: list[tuple[str, int | None]],
) -> None:
    input_exec.select_and_confirm(-1)
    assert calls == [("tap_left", 1), ("confirm", None)]
