"""Screenshot da janela do jogo via mss, salvando PNG em frames/."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import mss
from loguru import logger
from PIL import Image

from src.config import PATHS, WINDOW

_FILENAME_TS = "%Y%m%dT%H%M%S%f"


def _timestamp() -> str:
    return datetime.now().strftime(_FILENAME_TS)[:-3]


def grab(state: str | None = None) -> Path:
    """Captura a janela configurada e salva como PNG em frames/. Retorna o caminho."""
    PATHS.frames.mkdir(parents=True, exist_ok=True)
    region = {"left": WINDOW.x, "top": WINDOW.y, "width": WINDOW.w, "height": WINDOW.h}

    with mss.mss() as sct:
        shot = sct.grab(region)
        img = Image.frombytes("RGB", shot.size, shot.rgb)

    suffix = f"_{state}" if state else ""
    out_path = PATHS.frames / f"{_timestamp()}{suffix}.png"
    img.save(out_path, format="PNG")
    logger.debug("Frame salvo em {}", out_path)
    return out_path


def grab_many(n: int, interval_s: float) -> list[Path]:
    """Captura N frames espaçados por interval_s segundos."""
    if n <= 0:
        raise ValueError("n deve ser >= 1")
    paths: list[Path] = []
    for i in range(n):
        paths.append(grab())
        if i < n - 1:
            time.sleep(interval_s)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Captura screenshots da janela do jogo.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once", action="store_true", help="Captura 1 frame e sai.")
    group.add_argument("--n", type=int, metavar="N", help="Captura N frames espaçados.")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Intervalo em segundos entre frames com --n (default 1.0).",
    )
    args = parser.parse_args()

    if args.once:
        path = grab()
        print(path)
        return 0

    paths = grab_many(args.n, args.interval)
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
