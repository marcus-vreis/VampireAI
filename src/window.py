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
    foreground: bool = True
    hwnd: int = 0


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
    """Client area da janela do jogo. Cai no retângulo de config se não achar.

    `foreground` diz se a janela está em primeiro plano. Importa por dois motivos
    que se reforçam: `mss` captura uma REGIÃO DA TELA, não o conteúdo da janela,
    então qualquer coisa por cima do jogo entra no frame — já aconteceu de a
    página da Steam ser capturada e passar por combate. E o gamepad virtual só
    chega na janela focada, então agir sem foco seria inútil de qualquer jeito.
    """
    try:
        u32 = _user32()
        needle = title_substring.casefold()
        active = u32.GetForegroundWindow()
        for hwnd, title in _iter_top_level(u32):
            if needle in title.casefold():
                rect = _client_rect_on_screen(u32, hwnd)
                if rect.w > 0 and rect.h > 0:
                    return WindowLookup(
                        rect=rect, source="win32", foreground=hwnd == active, hwnd=hwnd
                    )
        logger.warning("Janela '{}' não encontrada — usando retângulo de config", title_substring)
    except (RuntimeError, OSError) as e:
        logger.warning("Lookup de janela falhou ({}) — usando retângulo de config", e)
    return WindowLookup(rect=WINDOW, source="config")


def warn_if_unfocused() -> bool:
    """Avisa quando o jogo não está em primeiro plano. Devolve se está.

    Para as ferramentas de linha de comando que emitem input de verdade. O
    agente já recusa agir sem foco (ADR-052), mas as CLIs não tinham essa
    proteção: `--scan-hand` aperta ← contra qualquer janela que estiver focada,
    e `--action confirm` aperta X. A contagem regressiva delas pede pra focar o
    jogo, mas pedir não é conferir.

    Avisa em vez de bloquear: quem roda uma CLI dessas está depurando de
    propósito, e às vezes quer ver o efeito noutro lugar.
    """
    found = find_game_window()
    if found.foreground:
        return True
    logger.warning(
        "O jogo NÃO está em primeiro plano. O input vai pra janela que estiver "
        "focada, não pro Vampire Crawlers."
    )
    return False


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
    print(
        f"source={found.source} x={r.x} y={r.y} w={r.w} h={r.h} "
        f"primeiro_plano={'sim' if found.foreground else 'NAO'}"
    )
    if not found.foreground:
        print(
            "  aviso: a janela do jogo não está em primeiro plano. A captura "
            "pega o que estiver por cima, e o gamepad não chega nela."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Pontos amostrados dentro da client area, em fração dela. O centro sozinho não
# basta: um balão de notificação no canto cobriria parte do frame e passaria.
_VISIBILITY_PROBES = ((0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))
_GA_ROOT = 2


def game_is_visible(lookup: WindowLookup | None = None) -> bool | None:
    """A janela do jogo está de fato NA FRENTE nos pixels que vão ser capturados?

    `mss` captura uma região da tela, então precisa haver alguma garantia de que
    o jogo não está coberto. O foco não serve pra isso na rotulagem: pra ler a
    tecla o terminal precisa do foco, então o jogo nunca está focado ali.

    A pergunta é respondida pelo **sistema**, não pelos pixels: `WindowFromPoint`
    diz qual janela está no topo em cada ponto. Isso substitui a checagem por
    conteúdo, que caía numa armadilha circular — ela exigia que a CV
    RECONHECESSE a tela, e as telas que mais precisam ser rotuladas são
    justamente as que a CV ainda não reconhece (título, menu, loja, game over).
    Com o guarda antigo, era impossível capturar o material necessário pra
    consertar o ponto cego.

    Devolve None quando não dá pra saber — sem janela localizada por Win32, ou
    fora do Windows. Quem chama decide o que fazer com a dúvida.
    """
    lookup = lookup or find_game_window()
    if lookup.source != "win32" or not lookup.hwnd:
        return None
    try:
        u32 = _user32()
        u32.WindowFromPoint.argtypes = [wintypes.POINT]
        u32.WindowFromPoint.restype = wintypes.HWND
        u32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        u32.GetAncestor.restype = wintypes.HWND
    except (RuntimeError, OSError):
        return None
    return all(_probe_belongs_to(u32, lookup, fx, fy) for fx, fy in _VISIBILITY_PROBES)


def _probe_belongs_to(u32, lookup: WindowLookup, fx: float, fy: float) -> bool:
    r = lookup.rect
    ponto = wintypes.POINT(int(r.x + fx * r.w), int(r.y + fy * r.h))
    topo = u32.WindowFromPoint(ponto)
    if not topo:
        return False
    raiz = u32.GetAncestor(topo, _GA_ROOT) or topo
    return int(raiz) == int(lookup.hwnd)
