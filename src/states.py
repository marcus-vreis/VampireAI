"""Enum dos estados do jogo + detector híbrido CV/VLM.

A detecção era 100% VLM: a chamada mais frequente do agente e a menos confiável.
Dos 39 frames que são o mapa, o modelo rotulou ao menos 9 como outra coisa — a
ponto de rodar um scan de cartas de 4 passos em cima do mapa.

Agora a CV resolve mapa e combate (que dominam o loop) em ~19ms e sem erro, e o
VLM só é chamado nas telas raras — com a lista de opções já restrita ao subgrupo
certo, o que também torna a pergunta mais fácil pra ele.
"""

from __future__ import annotations

import argparse
from enum import Enum

import cv2
from loguru import logger

from src.config import PATHS
from src.llm import ask_vlm
from src.schemas import StateDetection
from src.vision.screen import Verdict, signature


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
    DECK = "deck"  # tela "Baralho": o deck inteiro, aberto pelo jogador
    NOTICE = "notice"  # painel de aviso/confirmação: só texto e um botão


class NotTheGameError(RuntimeError):
    """A captura não pegou a janela do jogo — não há estado a inferir."""


_DIALOG_STATES = {
    GameState.LEVEL_UP,
    GameState.CHEST,
    GameState.BOSS_CHEST,
    GameState.CHEST_CARD_TARGET,
    GameState.SHOP,
    GameState.NOTICE,
}
_OTHER_STATES = {
    GameState.TITLE,
    GameState.MENU,
    GameState.GAME_OVER,
    GameState.STAGE_COMPLETE,
    GameState.GAME_COMPLETE,
    GameState.NOTICE,
}


def _ask_state(frame_path: str, prompt_file: str, allowed: set[GameState]) -> GameState:
    prompt = (PATHS.prompts / prompt_file).read_text(encoding="utf-8")
    result = ask_vlm(frame_path, prompt, schema=StateDetection)
    assert isinstance(result, dict)
    raw = str(result.get("estado", "")).strip().lower()
    try:
        state = GameState(raw)
    except ValueError as e:
        raise ValueError(f"VLM retornou estado desconhecido: {raw!r}") from e
    if state not in allowed:
        raise ValueError(f"VLM retornou {raw!r}, fora do subgrupo esperado")
    return state


def detect_state(frame_path: str) -> GameState:
    """Estado do jogo num frame. CV decide; VLM só desempata telas raras."""
    frame = cv2.imread(frame_path)
    if frame is None:
        raise FileNotFoundError(frame_path)

    sig = signature(frame)
    logger.debug(
        "assinatura: verdict={} parch={} slate={} cartas={} hud={}",
        sig.verdict.value, sig.parchment, sig.slate, sig.cards, sig.hud,
    )

    if sig.verdict is Verdict.NOT_GAME:
        raise NotTheGameError(
            f"a captura não parece ser o jogo (brilho={sig.brightness}). "
            "A janela do Vampire Crawlers está aberta e visível?"
        )
    if sig.verdict is Verdict.COMBAT:
        return GameState.COMBAT
    if sig.verdict is Verdict.MAP:
        return GameState.MAP
    if sig.verdict is Verdict.DECK:
        return GameState.DECK
    if sig.verdict is Verdict.DIALOG:
        return _ask_state(frame_path, "detect_dialog.txt", _DIALOG_STATES)
    return _ask_state(frame_path, "detect_other.txt", _OTHER_STATES)


def main() -> int:
    parser = argparse.ArgumentParser(description="Detecta o estado do jogo num frame.")
    parser.add_argument("--frame", required=True)
    parser.add_argument(
        "--cv-only", action="store_true", help="Só a assinatura de CV, sem chamar o VLM."
    )
    args = parser.parse_args()

    if args.cv_only:
        sig = signature(cv2.imread(args.frame))
        print(
            f"{sig.verdict.value}  parch={sig.parchment} slate={sig.slate} "
            f"cartas={sig.cards} hud={sig.hud} brilho={sig.brightness}"
        )
        return 0

    state = detect_state(args.frame)
    logger.info("Estado: {}", state.value)
    print(state.value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
