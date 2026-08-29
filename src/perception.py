"""Percepção do jogo: CV determinística primeiro, VLM só para semântica.

Antes, todo sinal vinha do VLM sobre um recorte da mão que estava errado — cobria
menos da metade do leque, cortando a carta da ponta e fatiando ao meio a
selecionada. Contagem e leitura de carta eram ruins por isso, e o consenso de 3
amostras (ADR-019) só tirava a média de uma leitura impossível.

Agora: cartas, cursor e estado saem de `src.vision`; o VLM lê o texto de UMA
carta grande e nítida, e só quando ela ainda não está no `CardDB`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image
from pydantic import BaseModel

from src import gamepad
from src.capture import grab
from src.carddb import CardDB, CardRecord
from src.config import GAMEPAD, PATHS
from src.llm import ask_vlm
from src.schemas import CardScanFrame, ChestState, LevelUpState, ShopState
from src.states import GameState, detect_state
from src.vision.cards import CostCircle, card_bbox, detect_card_slots
from src.vision.digits import GlyphBook
from src.vision.hud import ORB_BOX, orb_glyphs, read_hp, read_mana

# Duas leituras do cursor a menos de 12px são a mesma carta.
_SAME_CARD_PX = 12
_MAX_HAND = 12


class PerceivedFrame(BaseModel):
    state: GameState
    data: dict | None = None


class HandScan(BaseModel):
    cards: list[CardScanFrame]
    cursor_idx: int
    mana: int | None = None
    hp: tuple[int, int] | None = None  # (atual, máximo)

    @property
    def total(self) -> int:
        return len(self.cards)


def _load_prompt(filename: str) -> str:
    return (PATHS.prompts / filename).read_text(encoding="utf-8")


def _crop_card(frame: np.ndarray, circle: CostCircle) -> np.ndarray:
    x, y, w, h = card_bbox(circle)
    return frame[max(0, y) : y + h, max(0, x) : x + w]


def read_card(card_img: np.ndarray, db: CardDB | None = None) -> CardScanFrame:
    """Identidade de uma carta. Consulta o cache antes de gastar uma chamada VLM."""
    if db is not None:
        hit = db.lookup(card_img)
        if hit is not None:
            logger.debug("carta em cache: {}", hit.nome)
            return CardScanFrame(**vars(hit))

    pil = Image.fromarray(cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB))
    result = ask_vlm(None, _load_prompt("card_scan.txt"), schema=CardScanFrame, image=pil)
    assert isinstance(result, dict)
    card = CardScanFrame(**result)
    if db is not None and _worth_caching(card):
        db.remember(card_img, CardRecord(**result))
    return card


def _worth_caching(card: CardScanFrame) -> bool:
    """Só guarda leitura completa.

    Custo ilegível congelado em cache é pior que chamada repetida: `combat.validate`
    não consegue checar se a carta cabe na mana quando `mana` é None, então uma
    leitura ruim vira jogada ilegal em todo turno seguinte em que a carta aparece.
    """
    return bool(card.nome) and card.mana is not None


class _ManaResult(BaseModel):
    mana: int | None = None


def read_mana_hybrid(frame: np.ndarray, book: GlyphBook | None = None) -> int | None:
    """Mana pelo caminho mais barato disponível.

    Ordem: glifo já aprendido (microssegundos) → Tesseract (~5ms, se instalado) →
    modelo (~830ms). O que o modelo responde é ensinado ao livro de glifos, então
    a chamada some depois que cada algarismo aparece duas vezes. É o mesmo padrão
    do `CardDB`: o modelo ensina, a CV assume.
    """
    value = read_mana(frame, book)
    if value is not None:
        return value
    x0, y0, x1, y1 = ORB_BOX
    orb = frame[y0:y1, x0:x1]
    if orb.size == 0:
        return None
    pil = Image.fromarray(cv2.cvtColor(orb, cv2.COLOR_BGR2RGB))
    try:
        result = ask_vlm(
            None, _load_prompt("read_mana_orb.txt"), schema=_ManaResult, image=pil
        )
    except RuntimeError as e:
        logger.warning("Leitura de mana falhou: {}", e)
        return None
    assert isinstance(result, dict)
    value = result.get("mana")
    if value is not None and book is not None:
        book.teach(orb_glyphs(frame), value)
    return value


def _order_by_position(positions: list[int], final_x: int) -> tuple[list[int], int]:
    """Ordem esquerda→direita e o índice onde o cursor parou."""
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    ranked = [positions[i] for i in order]
    cursor = min(range(len(ranked)), key=lambda i: abs(ranked[i] - final_x))
    return order, cursor


def scan_combat_hand(
    db: CardDB | None = None, book: GlyphBook | None = None
) -> HandScan:
    """Percorre a mão com ← lendo uma carta por passo.

    Não precisa saber o total antes: a travessia termina sozinha quando o cursor
    volta a uma carta já vista (o jogo dá a volta) ou trava na ponta. Isso remove
    a dependência de uma contagem num único frame — imprecisa porque a carta
    selecionada, levantada, cobre o círculo de custo da vizinha à direita.
    """
    positions: list[int] = []
    cards: list[CardScanFrame] = []
    mana: int | None = None
    hp: tuple[int, int] | None = None
    final_x = 0

    for step in range(_MAX_HAND):
        frame = cv2.imread(str(grab(state=f"scan_{step}")))
        if mana is None:
            mana = read_mana_hybrid(frame, book)
        if hp is None:
            hp = read_hp(frame, book)
        selected = detect_card_slots(frame).selected
        if selected is None:
            logger.info("Sem carta destacada no passo {} — fim da travessia", step)
            break
        final_x = selected.x
        if any(abs(selected.x - p) < _SAME_CARD_PX for p in positions):
            break  # voltou a uma carta já lida: a mão inteira foi vista
        positions.append(selected.x)
        cards.append(read_card(_crop_card(frame, selected), db))
        logger.info("scan {}: {} (mana={})", len(cards), cards[-1].nome, cards[-1].mana)
        gamepad.tap_left(1)
        time.sleep(GAMEPAD.post_dpad_settle_s)

    if not positions:
        return HandScan(cards=[], cursor_idx=0, mana=mana, hp=hp)
    order, cursor = _order_by_position(positions, final_x)
    return HandScan(
        cards=[cards[i] for i in order], cursor_idx=cursor, mana=mana, hp=hp
    )


_PROMPT_BY_STATE: dict[GameState, tuple[str, type[BaseModel]]] = {
    GameState.LEVEL_UP: ("level_up.txt", LevelUpState),
    GameState.SHOP: ("shop.txt", ShopState),
    GameState.CHEST: ("chest.txt", ChestState),
    GameState.CHEST_CARD_TARGET: ("chest_card_target.txt", ChestState),
    GameState.BOSS_CHEST: ("chest.txt", ChestState),
}


def perceive(frame_path: str) -> PerceivedFrame:
    """Estado do frame e, para telas de escolha, o conteúdo lido pelo VLM.

    Combate e mapa não passam por aqui pra extrair conteúdo — os handlers usam
    `src.vision` direto, que é exato e ~90x mais rápido.
    """
    if not Path(frame_path).is_file():
        raise FileNotFoundError(frame_path)

    state = detect_state(frame_path)
    logger.debug("Estado detectado: {}", state.value)

    entry = _PROMPT_BY_STATE.get(state)
    if entry is None:
        return PerceivedFrame(state=state)

    prompt_file, schema = entry
    result = ask_vlm(frame_path, _load_prompt(prompt_file), schema=schema)
    assert isinstance(result, dict)
    return PerceivedFrame(state=state, data=result)


def default_carddb() -> CardDB:
    return CardDB(PATHS.notes / "cards.json")


def default_glyphbook() -> GlyphBook:
    return GlyphBook(PATHS.notes / "glyphs.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Percepção de um frame.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--frame", help="Caminho de PNG pra perceber.")
    group.add_argument("--cards", help="Só a detecção de cartas num PNG (sem VLM).")
    group.add_argument(
        "--scan-hand", action="store_true", help="Captura e percorre a mão em combate."
    )
    parser.add_argument("--countdown", type=int, default=3)
    args = parser.parse_args()

    if args.cards:
        slots = detect_card_slots(cv2.imread(args.cards))
        print(
            f"visiveis={slots.visible_total} cursor={slots.selected_idx} "
            f"circulos={[(c.x, c.side) for c in slots.circles]}"
        )
        return 0

    if args.frame:
        print(perceive(args.frame).model_dump_json(indent=2))
        return 0

    for s in range(args.countdown, 0, -1):
        logger.info("Capturando em {}s — foque o jogo (combate)...", s)
        time.sleep(1)

    scan = scan_combat_hand(default_carddb(), default_glyphbook())
    print(json.dumps(scan.model_dump(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
