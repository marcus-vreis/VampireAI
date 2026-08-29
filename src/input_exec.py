"""Ações de alto nível do jogo, traduzidas em sequências de gamepad.

Sem coords. Cartas/opções são selecionadas por travessia (esquerda/direita) com X
para confirmar. A "carta selecionada" é a maior/em destaque na tela.
"""

from __future__ import annotations

import argparse
import time

from loguru import logger

from src import gamepad
from src.config import GAMEPAD


def navigate_horizontal(steps: int) -> None:
    """Navega N passos: positivo = direita, negativo = esquerda, 0 = noop."""
    if steps == 0:
        return
    if steps > 0:
        gamepad.tap_right(steps)
    else:
        gamepad.tap_left(-steps)


def select_and_confirm(steps: int) -> None:
    """Move N passos no eixo horizontal e aperta X."""
    navigate_horizontal(steps)
    time.sleep(GAMEPAD.post_dpad_settle_s)
    gamepad.confirm()


def end_turn() -> None:
    """Encerra o turno via Bola (Circle). Idealmente, jogar todas as cartas que
    couberem na mana antes — o jogo finaliza turno automaticamente quando
    a mão acaba ou nada mais é pagável."""
    gamepad.skip()


def confirm() -> None:
    gamepad.confirm()


def cancel() -> None:
    gamepad.cancel()


def turn_left() -> None:
    gamepad.turn_left()


def turn_right() -> None:
    gamepad.turn_right()


def walk_forward() -> None:
    gamepad.walk_forward()


def walk_back() -> None:
    gamepad.walk_back()


def main() -> int:
    parser = argparse.ArgumentParser(description="Ações de alto nível via gamepad.")
    parser.add_argument("--action", required=True, choices=[
        "confirm", "cancel", "skip", "turn_left", "turn_right",
        "walk_forward", "walk_back", "end_turn",
    ])
    parser.add_argument("--countdown", type=int, default=3)
    args = parser.parse_args()

    for s in range(args.countdown, 0, -1):
        logger.info("Executando {} em {}s — foque o jogo agora...", args.action, s)
        time.sleep(1)

    {
        "confirm": confirm,
        "cancel": cancel,
        "skip": gamepad.skip,
        "turn_left": turn_left,
        "turn_right": turn_right,
        "walk_forward": walk_forward,
        "walk_back": walk_back,
        "end_turn": end_turn,
    }[args.action]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
