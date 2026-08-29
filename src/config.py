"""Coordenadas, paths, timing de gamepad e constantes do projeto."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mss
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.example", override=False)


def _env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value is not None and value != "" else default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_int_or_auto(key: str, auto_value: int) -> int:
    raw = os.getenv(key, "").strip().lower()
    if raw in ("", "auto"):
        return auto_value
    return int(raw)


def _primary_monitor_rect() -> tuple[int, int, int, int]:
    with mss.mss() as sct:
        for mon in sct.monitors[1:]:
            if mon.get("is_primary"):
                return mon["left"], mon["top"], mon["width"], mon["height"]
        m = sct.monitors[1]
        return m["left"], m["top"], m["width"], m["height"]


@dataclass(frozen=True)
class WindowRect:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class LLMConfig:
    """Dois modelos: um enxerga, outro raciocina.

    Depois que a CV assumiu a percepção geométrica, as chamadas ao modelo se
    separaram em duas naturezas. Transcrever uma carta precisa de visão. Escolher
    a jogada e a recompensa é texto puro — essas chamadas já passavam `image=None`
    para um VLM, onde a torre visual não contribui nada e o backbone de linguagem
    é mais fraco que o de um modelo de texto do mesmo tamanho.

    `text_model` cai em `vision_model` quando não configurado, então uma
    instalação existente continua funcionando sem mudar nada.
    """

    base_url: str
    api_key: str
    vision_model: str
    text_model: str
    timeout_s: float
    max_retries: int
    backoff_base_s: float
    image_max_side: int

    def pick(self, has_image: bool) -> str:
        return self.vision_model if has_image else self.text_model


@dataclass(frozen=True)
class GamepadConfig:
    press_hold_s: float
    between_actions_s: float
    boot_delay_s: float
    post_dpad_settle_s: float


@dataclass(frozen=True)
class Paths:
    root: Path
    frames: Path
    logs: Path
    notes: Path
    prompts: Path

    def ensure(self) -> None:
        for p in (self.frames, self.logs, self.notes):
            p.mkdir(parents=True, exist_ok=True)


def _build_paths() -> Paths:
    return Paths(
        root=PROJECT_ROOT,
        frames=PROJECT_ROOT / _env_str("FRAMES_DIR", "frames"),
        logs=PROJECT_ROOT / _env_str("LOGS_DIR", "logs"),
        notes=PROJECT_ROOT / _env_str("NOTES_DIR", "notes"),
        prompts=PROJECT_ROOT / "src" / "prompts",
    )


def _build_window() -> WindowRect:
    w = _env_int("GAME_WINDOW_W", 1280)
    h = _env_int("GAME_WINDOW_H", 720)
    mon_left, mon_top, mon_w, mon_h = _primary_monitor_rect()
    auto_x = mon_left + (mon_w - w) // 2
    auto_y = mon_top + (mon_h - h) // 2
    return WindowRect(
        x=_env_int_or_auto("GAME_WINDOW_X", auto_x),
        y=_env_int_or_auto("GAME_WINDOW_Y", auto_y),
        w=w,
        h=h,
    )


def _build_llm() -> LLMConfig:
    vision = _env_str("VLM_MODEL", "qwen2.5vl:7b")
    return LLMConfig(
        base_url=_env_str("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key=_env_str("OLLAMA_API_KEY", "ollama"),
        vision_model=vision,
        text_model=_env_str("TEXT_MODEL", vision),
        timeout_s=_env_float("LLM_TIMEOUT_SECONDS", 60.0),
        max_retries=_env_int("LLM_MAX_RETRIES", 3),
        backoff_base_s=_env_float("LLM_BACKOFF_BASE_SECONDS", 1.5),
        image_max_side=_env_int("VLM_IMAGE_MAX_SIDE", 768),
    )


def _build_gamepad() -> GamepadConfig:
    # Ver jogo.md: ~1s entre ações compostas evita comer input.
    return GamepadConfig(
        press_hold_s=_env_float("GAMEPAD_PRESS_HOLD_S", 0.08),
        between_actions_s=_env_float("GAMEPAD_BETWEEN_ACTIONS_S", 0.25),
        boot_delay_s=_env_float("GAMEPAD_BOOT_DELAY_S", 0.5),
        post_dpad_settle_s=_env_float("GAMEPAD_POST_DPAD_SETTLE_S", 0.4),
    )


PATHS = _build_paths()
WINDOW = _build_window()
LLM = _build_llm()
GAMEPAD = _build_gamepad()

LLM_LOG_FILE = PATHS.logs / "llm.jsonl"

# Regiões de UI vivem em `src/vision/regions.py`, medidas contra o client area que
# `src/window.py` localiza. As que existiam aqui foram removidas na ADR-022: a
# antiga `hand_area = (380, 460, 480, 260)` cobria menos da metade do leque e era
# a causa raiz da contagem e da leitura de carta erradas — não vale deixar à mão
# pra ser reusada por engano. Limiares de OCR estão em `src/vision/hud.py`.
