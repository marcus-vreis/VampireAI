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
    base_url: str
    api_key: str
    model: str
    timeout_s: float
    max_retries: int
    backoff_base_s: float
    image_max_side: int


@dataclass(frozen=True)
class GamepadConfig:
    press_hold_s: float
    between_actions_s: float
    boot_delay_s: float
    post_dpad_settle_s: float


@dataclass(frozen=True)
class PerceptionConfig:
    """Tunables da percepção (especialmente contagem de cartas)."""
    count_samples: int  # k-vote: quantas chamadas VLM por contagem (mode wins)
    enhance_contrast: float  # 1.0 = sem mudança; >1 amplifica
    enhance_saturation: float  # idem; ajuda destacar bolinhas azuis
    enhance_sharpness: float  # idem; bordas de números/bolinhas


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
    return LLMConfig(
        base_url=_env_str("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key=_env_str("OLLAMA_API_KEY", "ollama"),
        model=_env_str("VLM_MODEL", "qwen2.5vl:7b"),
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


def _build_perception() -> PerceptionConfig:
    return PerceptionConfig(
        count_samples=max(1, _env_int("PERCEPTION_COUNT_SAMPLES", 3)),
        enhance_contrast=_env_float("PERCEPTION_ENHANCE_CONTRAST", 1.25),
        enhance_saturation=_env_float("PERCEPTION_ENHANCE_SATURATION", 1.4),
        enhance_sharpness=_env_float("PERCEPTION_ENHANCE_SHARPNESS", 1.2),
    )


PATHS = _build_paths()
WINDOW = _build_window()
LLM = _build_llm()
GAMEPAD = _build_gamepad()
PERCEPTION = _build_perception()

LLM_LOG_FILE = PATHS.logs / "llm.jsonl"


def is_region_set(region: tuple[int, int, int, int]) -> bool:
    return all(v >= 0 for v in region) and region[2] > 0 and region[3] > 0


# Regiões para CROPS de UI antes de mandar pro VLM. (x, y, w, h) relativos à janela do jogo (1280x720).
# Estratégia: recortar pequenas áreas focadas → VLM 7B fica muito mais preciso.
# Estimativas baseadas em screenshot real de combate. Refinar com `python -m src.perception --crop NAME`.
UI_REGIONS: dict[str, tuple[int, int, int, int]] = {
    # Mão: leque inferior central com cartas (4-7 cartas)
    "hand_area": (380, 460, 480, 260),
    # Orbe de mana: bola azul grande no canto inferior direito (acima de "Finalizar turno")
    "mana_orb": (1000, 470, 140, 140),
    # Coração de HP: canto inferior esquerdo
    "hp_heart": (140, 470, 140, 140),
    # Stats topo-esquerdo (ouro, vida%, crit%, dmg)
    "stats_top_left": (15, 50, 220, 130),
    # Stats topo-direito (probabilidades de fase)
    "stats_top_right": (1080, 50, 200, 80),
}

# Regiões para OCR (pytesseract). Mantemos vazio por enquanto — VLM via crops já cobre.
OCR_REGIONS: dict[str, tuple[int, int, int, int]] = {
    "hp_player": (-1, -1, -1, -1),
    "mana": (-1, -1, -1, -1),
}

OCR_UPSCALE = 4
OCR_THRESHOLD = 140
