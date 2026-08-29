"""Virtual DualShock gamepad via vgamepad/ViGEm. Mapeia ações do jogo para botões."""

from __future__ import annotations

import argparse
import time
from enum import Enum

import vgamepad as vg
from loguru import logger

from src.config import GAMEPAD


class Button(str, Enum):
    DPAD_UP = "dpad_up"
    DPAD_DOWN = "dpad_down"
    DPAD_LEFT = "dpad_left"
    DPAD_RIGHT = "dpad_right"
    CONFIRM = "confirm"  # X (cross)
    CANCEL = "cancel"  # square
    SKIP = "skip"  # circle — pula turno em combate
    TURN_LEFT = "turn_left"  # L2
    TURN_RIGHT = "turn_right"  # R2


_BUTTON_MAP = {
    Button.CONFIRM: vg.DS4_BUTTONS.DS4_BUTTON_CROSS,
    Button.CANCEL: vg.DS4_BUTTONS.DS4_BUTTON_SQUARE,
    Button.SKIP: vg.DS4_BUTTONS.DS4_BUTTON_CIRCLE,
}

_DPAD_MAP = {
    Button.DPAD_UP: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NORTH,
    Button.DPAD_DOWN: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_SOUTH,
    Button.DPAD_LEFT: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_WEST,
    Button.DPAD_RIGHT: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_EAST,
}

_pad: vg.VDS4Gamepad | None = None

# Trava de emissão. Existe porque garantir "não mandar input" no chamador não
# funciona: o replay tentou trocar `input_exec` por um gravador e mesmo assim
# saiu input, porque `agent._NUDGE_ACTION` já tinha as funções ligadas desde o
# import. A única garantia confiável é aqui embaixo, antes do driver.
_dry_run = False


def set_dry_run(enabled: bool) -> None:
    """Liga/desliga a emissão real de input. Ligado = nada chega ao driver."""
    global _dry_run
    _dry_run = enabled
    if enabled:
        logger.info("Gamepad em dry-run: nenhum input será emitido")


def is_dry_run() -> bool:
    return _dry_run


def _get_pad() -> vg.VDS4Gamepad:
    global _pad
    if _pad is None:
        logger.info("Inicializando virtual DualShock 4 (ViGEm Bus driver)")
        _pad = vg.VDS4Gamepad()
        time.sleep(GAMEPAD.boot_delay_s)
    return _pad


def press(button: Button, hold_s: float | None = None) -> None:
    """Aperta e solta um botão, com hold_s entre press e release.

    Em dry-run sai antes de tocar no driver — inclusive antes de criar o
    dispositivo virtual, que `_get_pad` faria.
    """
    hold = hold_s if hold_s is not None else GAMEPAD.press_hold_s
    if _dry_run:
        logger.debug("[dry-run] press {}", button.value)
        return

    pad = _get_pad()

    if button in _DPAD_MAP:
        pad.directional_pad(direction=_DPAD_MAP[button])
        pad.update()
        time.sleep(hold)
        pad.directional_pad(direction=vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NONE)
        pad.update()
    elif button is Button.TURN_LEFT:
        pad.left_trigger(value=255)
        pad.update()
        time.sleep(hold)
        pad.left_trigger(value=0)
        pad.update()
    elif button is Button.TURN_RIGHT:
        pad.right_trigger(value=255)
        pad.update()
        time.sleep(hold)
        pad.right_trigger(value=0)
        pad.update()
    else:
        ds4_btn = _BUTTON_MAP[button]
        pad.press_button(button=ds4_btn)
        pad.update()
        time.sleep(hold)
        pad.release_button(button=ds4_btn)
        pad.update()

    logger.debug("press {} hold={}s", button.value, hold)
    time.sleep(GAMEPAD.between_actions_s)


def tap_left(times: int = 1) -> None:
    for _ in range(times):
        press(Button.DPAD_LEFT)


def tap_right(times: int = 1) -> None:
    for _ in range(times):
        press(Button.DPAD_RIGHT)


def confirm() -> None:
    press(Button.CONFIRM)


def cancel() -> None:
    press(Button.CANCEL)


def skip() -> None:
    """Bola/Circle — pula turno em combate, ou cancela em outras telas."""
    press(Button.SKIP)


def turn_left() -> None:
    press(Button.TURN_LEFT)


def turn_right() -> None:
    press(Button.TURN_RIGHT)


def walk_forward() -> None:
    press(Button.DPAD_UP)


def walk_back() -> None:
    press(Button.DPAD_DOWN)


def walk_left() -> None:
    press(Button.DPAD_LEFT)


def walk_right() -> None:
    press(Button.DPAD_RIGHT)


def reset() -> None:
    """Solta tudo. Chamar antes de abortar.

    Em dry-run não há dispositivo pra resetar — e criar um só pra resetar seria
    exatamente o que a trava existe pra evitar.
    """
    if _dry_run or _pad is None:
        return
    pad = _get_pad()
    pad.reset()
    pad.update()


def _self_test(seconds: float = 1.0) -> None:
    """Aperta cada botão sequencialmente para validar mapeamento no jogo."""
    sequence: list[Button] = [
        Button.DPAD_RIGHT, Button.DPAD_LEFT, Button.DPAD_UP, Button.DPAD_DOWN,
        Button.TURN_RIGHT, Button.TURN_LEFT, Button.CONFIRM, Button.CANCEL, Button.SKIP,
    ]
    logger.info("Self-test do gamepad — foque a janela do jogo agora.")
    time.sleep(3)
    for b in sequence:
        logger.info("→ {}", b.value)
        press(b)
        time.sleep(seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gamepad virtual via vgamepad.")
    parser.add_argument("--test", action="store_true", help="Aperta cada botão em sequência.")
    parser.add_argument(
        "--press",
        choices=[b.value for b in Button],
        help="Aperta UM botão e sai.",
    )
    args = parser.parse_args()

    if args.test:
        _self_test()
        return 0
    if args.press:
        press(Button(args.press))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
