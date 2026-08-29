"""Localiza a área de cliente da janela do jogo via Win32.

Motivo: regiões de UI absolutas escorregam entre sessões quando a captura é um
retângulo calculado (barra de título presente ou não, janela movida). Capturar o
*client area* real elimina a classe inteira de erro na origem.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass

from loguru import logger

from src.config import WINDOW, WindowRect

_TITLE_SUBSTRING = "Vampire Crawlers"


@dataclass(frozen=True)
class WindowLookup:
    rect: WindowRect
    source: str  # "win32" | "config"


def _user32():
    if sys.platform != "win32":
        raise RuntimeError("Lookup de janela só existe no Windows")
    return ctypes.WinDLL("user32", use_last_error=True)


def _iter_top_level(u32) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _lparam):
        if not u32.IsWindowVisible(hwnd):
            return True
        length = u32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        u32.GetWindowTextW(hwnd, buf, length + 1)
        found.append((hwnd, buf.value))
        return True

    u32.EnumWindows(proto(cb), 0)
    return found


def _client_rect_on_screen(u32, hwnd: int) -> WindowRect:
    rect = wintypes.RECT()
    if not u32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise OSError(ctypes.get_last_error())
    origin = wintypes.POINT(0, 0)
    if not u32.ClientToScreen(hwnd, ctypes.byref(origin)):
        raise OSError(ctypes.get_last_error())
    return WindowRect(x=origin.x, y=origin.y, w=rect.right, h=rect.bottom)


def find_game_window(title_substring: str = _TITLE_SUBSTRING) -> WindowLookup:
    """Client area da janela do jogo. Cai no retângulo de config se não achar."""
    try:
        u32 = _user32()
        needle = title_substring.casefold()
        for hwnd, title in _iter_top_level(u32):
            if needle in title.casefold():
                rect = _client_rect_on_screen(u32, hwnd)
                if rect.w > 0 and rect.h > 0:
                    return WindowLookup(rect=rect, source="win32")
        logger.warning("Janela '{}' não encontrada — usando retângulo de config", title_substring)
    except (RuntimeError, OSError) as e:
        logger.warning("Lookup de janela falhou ({}) — usando retângulo de config", e)
    return WindowLookup(rect=WINDOW, source="config")


def main() -> int:
    parser = argparse.ArgumentParser(description="Localiza a janela do jogo.")
    parser.add_argument("--title", default=_TITLE_SUBSTRING)
    parser.add_argument("--list", action="store_true", help="Lista janelas visíveis.")
    args = parser.parse_args()

    if args.list:
        for hwnd, title in _iter_top_level(_user32()):
            print(f"{hwnd:>10} {title}")
        return 0

    found = find_game_window(args.title)
    r = found.rect
    print(f"source={found.source} x={r.x} y={r.y} w={r.w} h={r.h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
