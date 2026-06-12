"""Percepção do jogo: crops de UI focados → VLM 7B fica preciso.

Estratégia: em vez de mandar o frame inteiro pro VLM com prompt longo, recorta
áreas pequenas (mão, orbe de mana, coração) e usa prompts curtinhos por área.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from loguru import logger
from PIL import Image, ImageEnhance
from pydantic import BaseModel

from src import gamepad
from src.capture import grab
from src.config import GAMEPAD, PATHS, PERCEPTION, UI_REGIONS, is_region_set
from src.llm import ask_vlm
from src.schemas import (
    CardScanFrame,
    ChestState,
    LevelUpState,
    MapDirection,
    ShopState,
)
from src.states import GameState, detect_state


class _CountResult(BaseModel):
    total: int
    indice_selecionada: int | None = None


class _ManaResult(BaseModel):
    mana: int | None = None


class _HpResult(BaseModel):
    hp: int | None = None
    hp_max: int | None = None


class PerceivedFrame(BaseModel):
    state: GameState
    data: dict | None = None
    ocr: dict[str, int] = {}


def _load_prompt(filename: str) -> str:
    return (PATHS.prompts / filename).read_text(encoding="utf-8")


def _crop_region(frame_path: str, region_name: str) -> Image.Image:
    """Recorta uma região nomeada de UI_REGIONS de um frame."""
    region = UI_REGIONS.get(region_name)
    if region is None:
        raise KeyError(f"Região '{region_name}' não existe em UI_REGIONS")
    if not is_region_set(region):
        raise RuntimeError(f"Região '{region_name}' não calibrada (TBD)")
    x, y, w, h = region
    with Image.open(frame_path) as img:
        return img.convert("RGB").crop((x, y, x + w, y + h))


def enhance_for_count(img: Image.Image) -> Image.Image:
    """Realça contraste/saturação/nitidez para destacar bolinhas azuis de mana.

    O Qwen2.5-VL 7B confunde bolinhas pequenas em fundo embaçado. Pré-processar a
    imagem aumenta a separação das pipezinhas azuis das cartas — ajuda
    significativamente na contagem.
    """
    if PERCEPTION.enhance_contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(PERCEPTION.enhance_contrast)
    if PERCEPTION.enhance_saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(PERCEPTION.enhance_saturation)
    if PERCEPTION.enhance_sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(PERCEPTION.enhance_sharpness)
    return img


def _vote_count(samples: list[_CountResult]) -> _CountResult:
    """Mode dos totais; idx do maior total acordado, senão o mais comum não-null."""
    if not samples:
        return _CountResult(total=0, indice_selecionada=None)
    totals = Counter(s.total for s in samples)
    total, _ = totals.most_common(1)[0]
    idxs = [s.indice_selecionada for s in samples if s.total == total]
    nn = [i for i in idxs if i is not None]
    chosen_idx: int | None
    if nn:
        chosen_idx, _ = Counter(nn).most_common(1)[0]
    else:
        chosen_idx = None
    return _CountResult(total=total, indice_selecionada=chosen_idx)


def _count_cards(crop: Image.Image) -> _CountResult:
    """Roda `PERCEPTION.count_samples` chamadas e devolve o consenso (mode)."""
    crop = enhance_for_count(crop)
    prompt = _load_prompt("count_cards.txt")
    samples: list[_CountResult] = []
    for i in range(PERCEPTION.count_samples):
        try:
            result = ask_vlm(None, prompt, schema=_CountResult, image=crop)
            assert isinstance(result, dict)
            samples.append(_CountResult(**result))
        except Exception as e:  # noqa: BLE001
            logger.warning("count_cards amostra {}/{} falhou: {}", i + 1, PERCEPTION.count_samples, e)
    voted = _vote_count(samples)
    if len(samples) > 1:
        logger.info(
            "count_cards votos={} → total={} idx={}",
            [(s.total, s.indice_selecionada) for s in samples],
            voted.total,
            voted.indice_selecionada,
        )
    return voted


def _perceive_combat(frame_path: str) -> dict:
    """Combate via crops: hand_area pra contar cartas + mana_orb pra ler mana."""
    out: dict = {"total_cartas_visiveis": 0, "indice_selecionada": None, "mana": None}

    try:
        hand_crop = _crop_region(frame_path, "hand_area")
        voted = _count_cards(hand_crop)
        out["total_cartas_visiveis"] = voted.total
        out["indice_selecionada"] = voted.indice_selecionada
    except Exception as e:
        logger.warning("Falha em count_cards: {}", e)

    try:
        mana_crop = _crop_region(frame_path, "mana_orb")
        prompt = _load_prompt("read_mana_orb.txt")
        result = ask_vlm(None, prompt, schema=_ManaResult, image=mana_crop)
        assert isinstance(result, dict)
        out["mana"] = result["mana"]
        logger.debug("mana_orb: {}", result["mana"])
    except Exception as e:
        logger.warning("Falha em read_mana_orb: {}", e)

    return out


_PROMPT_BY_STATE: dict[GameState, tuple[str, type[BaseModel] | None]] = {
    GameState.LEVEL_UP: ("level_up.txt", LevelUpState),
    GameState.MAP: ("map.txt", MapDirection),
    GameState.SHOP: ("shop.txt", ShopState),
    GameState.CHEST: ("chest.txt", ChestState),
    GameState.CHEST_CARD_TARGET: ("chest_card_target.txt", ChestState),
    GameState.BOSS_CHEST: ("chest.txt", ChestState),
    GameState.TITLE: (None, None),
    GameState.MENU: (None, None),
    GameState.GAME_OVER: (None, None),
    GameState.STAGE_COMPLETE: (None, None),
    GameState.GAME_COMPLETE: (None, None),
}


def perceive(frame_path: str) -> PerceivedFrame:
    """Detecta estado e extrai conteúdo. Combate usa crops; outros estados usam frame inteiro."""
    if not Path(frame_path).is_file():
        raise FileNotFoundError(frame_path)

    state = detect_state(frame_path)
    logger.debug("Estado detectado: {}", state.value)

    if state is GameState.COMBAT:
        data = _perceive_combat(frame_path)
        return PerceivedFrame(state=state, data=data)

    prompt_file, schema = _PROMPT_BY_STATE.get(state, (None, None))
    if prompt_file is None or schema is None:
        return PerceivedFrame(state=state)

    prompt = _load_prompt(prompt_file)
    result = ask_vlm(frame_path, prompt, schema=schema)
    assert isinstance(result, dict)
    return PerceivedFrame(state=state, data=result)


def scan_combat_hand(initial_frame: str, total_cards: int) -> list[CardScanFrame]:
    """Scan sequencial: cursor inicia mais à direita, captura, ←, captura, repete."""
    if total_cards <= 0:
        return []

    prompt = _load_prompt("card_scan.txt")
    cards: list[CardScanFrame] = []

    for i in range(total_cards):
        frame = initial_frame if i == 0 else str(grab(state=f"card_scan_{i}"))
        # Recorta só a mão pra dar mais contexto na carta destacada.
        try:
            hand_crop = enhance_for_count(_crop_region(frame, "hand_area"))
            result = ask_vlm(None, prompt, schema=CardScanFrame, image=hand_crop)
        except Exception:
            result = ask_vlm(frame, prompt, schema=CardScanFrame)
        assert isinstance(result, dict)
        cards.append(CardScanFrame(**result))
        logger.info("scan {}/{}: {} (mana={})", i + 1, total_cards, cards[-1].nome, cards[-1].mana)

        if i < total_cards - 1:
            gamepad.tap_left(1)
            time.sleep(GAMEPAD.post_dpad_settle_s)

    return cards


def main() -> int:
    parser = argparse.ArgumentParser(description="Percepção de um frame.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--frame", help="Caminho de PNG pra perceber.")
    group.add_argument(
        "--scan-hand", action="store_true",
        help="Captura agora e roda scan sequencial da mão em combate.",
    )
    group.add_argument(
        "--crop", metavar="REGION",
        help="Salva o crop nomeado (de UI_REGIONS) em frames/. Use junto com --frame.",
    )
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument(
        "--enhance", action="store_true",
        help="Aplica realce de contraste/saturação ao crop (mesmo do pipeline real).",
    )
    args = parser.parse_args()

    if args.crop:
        if not args.frame:
            logger.error("--crop precisa de --frame FRAME.png")
            return 2
        crop = _crop_region(args.frame, args.crop)
        if args.enhance:
            crop = enhance_for_count(crop)
        suffix = "_enhanced" if args.enhance else ""
        out = PATHS.frames / f"crop_{args.crop}{suffix}.png"
        crop.save(out)
        print(out)
        return 0

    if args.frame:
        perceived = perceive(args.frame)
        print(perceived.model_dump_json(indent=2))
        return 0

    for s in range(args.countdown, 0, -1):
        logger.info("Capturando em {}s — foque o jogo (combate)...", s)
        time.sleep(1)

    initial_frame = str(grab(state="combat_initial"))
    perceived = perceive(initial_frame)
    if perceived.state is not GameState.COMBAT:
        logger.error("Estado={}, esperado combat", perceived.state)
        return 1

    data = perceived.data or {}
    total = int(data.get("total_cartas_visiveis", 0) or 0)
    logger.info("Combate: {} cartas, idx={}, mana={}",
                total, data.get("indice_selecionada"), data.get("mana"))
    if total <= 0:
        return 1

    cards = scan_combat_hand(initial_frame, total)
    import json as _json
    print(_json.dumps([c.model_dump() for c in cards], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
