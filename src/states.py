"""Enum dos estados do jogo + detector via VLM."""

from __future__ import annotations

import argparse
from enum import Enum

from loguru import logger

from src.config import PATHS
from src.llm import ask_vlm
from src.schemas import StateDetection

_DETECT_PROMPT_FILE = "detect_state.txt"


class GameState(str, Enum):
    COMBAT = "combat"
    MAP = "map"
    LEVEL_UP = "level_up"
    SHOP = "shop"
    CHEST = "chest"
    CHEST_CARD_TARGET = "chest_card_target"
    BOSS_CHEST = "boss_chest"
    STAGE_COMPLETE = "stage_complete"
    GAME_COMPLETE = "game_complete"
    TITLE = "title"
    MENU = "menu"
    GAME_OVER = "game_over"


def _load_prompt() -> str:
    return (PATHS.prompts / _DETECT_PROMPT_FILE).read_text(encoding="utf-8")


def detect_state(frame_path: str) -> GameState:
    prompt = _load_prompt()
    result = ask_vlm(frame_path, prompt, schema=StateDetection)
    assert isinstance(result, dict)
    raw = str(result.get("estado", "")).strip().lower()
    try:
        return GameState(raw)
    except ValueError as e:
        raise ValueError(f"VLM retornou estado desconhecido: {raw!r}") from e


def main() -> int:
    parser = argparse.ArgumentParser(description="Detecta o estado do jogo num frame.")
    parser.add_argument("--frame", required=True)
    args = parser.parse_args()
    state = detect_state(args.frame)
    logger.info("Estado: {}", state.value)
    print(state.value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
