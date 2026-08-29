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
from statistics import median

import cv2
import numpy as np
from loguru import logger
from PIL import Image
from pydantic import BaseModel

from src import gamepad
from src.capture import grab
from src.carddb import CardDB, CardRecord
from src.config import PATHS
from src.llm import ask_vlm
from src.schemas import CardScanFrame, ChestState, LevelUpState, ShopState
from src.states import GameState, detect_state
from src.vision.cards import (
    CostCircle,
    card_bbox,
    detect_card_slots,
    detect_choice_slots,
)
from src.vision.digits import GlyphBook
from src.vision.hud import HEART_BOX, ORB_BOX, heart_rows, orb_glyphs, read_hp, read_mana

# Duas leituras do cursor a menos de 12px são a mesma carta.
_SAME_CARD_PX = 12
_MAX_HAND = 12
# Teto de espera pelo cursor sair do lugar depois de um ←. Só é atingido quando
# o cursor realmente não se move (ponta do leque), e aí a travessia termina.
_CURSOR_MOVE_TIMEOUT_S = 0.8


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


class _HpResult(BaseModel):
    hp: int | None = None
    hp_max: int | None = None


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


def read_hp_hybrid(
    frame: np.ndarray, book: GlyphBook | None = None
) -> tuple[int, int] | None:
    """HP pelo livro de glifos, caindo pro modelo e ensinando o que ele responder.

    Sem este caminho o HP nunca era lido: `read_hp` só consulta o livro, e nada
    ensinava os algarismos do coração — então o dado que a ADR-036 coloca no
    prompt de combate ficava sempre ausente.
    """
    known = read_hp(frame, book)
    if known is not None:
        return known

    rows = heart_rows(frame)
    x0, y0, x1, y1 = HEART_BOX
    heart = frame[y0:y1, x0:x1]
    if len(rows) != 2 or heart.size == 0:
        return None
    pil = Image.fromarray(cv2.cvtColor(heart, cv2.COLOR_BGR2RGB))
    try:
        result = ask_vlm(
            None, _load_prompt("read_hp_heart.txt"), schema=_HpResult, image=pil
        )
    except RuntimeError as e:
        logger.warning("Leitura de HP falhou: {}", e)
        return None
    assert isinstance(result, dict)
    current, maximum = result.get("hp"), result.get("hp_max")
    if current is None or maximum is None:
        return None
    if book is not None:
        book.teach(rows[0], current)
        book.teach(rows[1], maximum)
    return current, maximum


def _tap_and_wait(previous_x: int, left: bool = True) -> tuple[np.ndarray, CostCircle | None]:
    """Aperta ← (ou →) e captura até o cursor sair do lugar. Devolve o frame estável.

    Substitui um `sleep` fixo de 400ms que respondia por 81% do custo de cada
    passo da travessia. Esperar o EFEITO em vez de um tempo arbitrário é mais
    rápido quando o jogo responde logo e mais seguro quando ele demora — o sleep
    cego podia ler um frame ainda em animação.
    """
    (gamepad.tap_left if left else gamepad.tap_right)(1)
    deadline = time.monotonic() + _CURSOR_MOVE_TIMEOUT_S
    frame, selected = None, None
    while True:
        frame = cv2.imread(str(grab(state="scan")))
        selected = detect_card_slots(frame).selected
        moved = selected is not None and abs(selected.x - previous_x) >= _SAME_CARD_PX
        if moved or time.monotonic() >= deadline:
            return frame, selected


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

    frame = cv2.imread(str(grab(state="scan_0")))
    selected = detect_card_slots(frame).selected
    for step in range(_MAX_HAND):
        if mana is None:
            mana = read_mana_hybrid(frame, book)
        if hp is None:
            hp = read_hp_hybrid(frame, book)
        if selected is None:
            logger.info("Sem carta destacada no passo {} — fim da travessia", step)
            break
        final_x = selected.x
        if any(abs(selected.x - p) < _SAME_CARD_PX for p in positions):
            break  # voltou a uma carta já lida: a mão inteira foi vista
        positions.append(selected.x)
        cards.append(_read_with_retry(_crop_card(frame, selected), db))
        logger.info("scan {}: {} (mana={})", len(cards), cards[-1].nome, cards[-1].mana)
        frame, selected = _tap_and_wait(final_x)

    if not positions:
        return HandScan(cards=[], cursor_idx=0, mana=mana, hp=hp)
    order, cursor = _order_by_position(positions, final_x)
    return HandScan(
        cards=[cards[i] for i in order], cursor_idx=cursor, mana=mana, hp=hp
    )


def read_hud(book: GlyphBook | None = None) -> tuple[int | None, tuple[int, int] | None]:
    """Captura e lê só mana e HP. ~95ms, contra ~2.5s de uma travessia."""
    frame = cv2.imread(str(grab(state="hud")))
    return read_mana_hybrid(frame, book), read_hp_hybrid(frame, book)


def _identify(frame: np.ndarray, circle: CostCircle, db: CardDB | None) -> str | None:
    """Nome da carta destacada pelo cache. None se ela não é conhecida."""
    if db is None:
        return None
    hit = db.lookup(_crop_card(frame, circle))
    return hit.nome if hit is not None else None


def seek_card(
    target: int, hand: list[CardScanFrame], db: CardDB | None, max_steps: int = _MAX_HAND
) -> bool:
    """Move o cursor até a carta destacada SER a desejada. True se chegou.

    Posiciona por identidade, não por aritmética de índice. O agente antes
    calculava `alvo - cursor`, navegava e apertava X **sem conferir onde o cursor
    parou** — com o índice errado, jogava a carta errada em silêncio.

    Cartas de mesmo nome são intercambiáveis: se a mão tem dois "Tomo Vazio",
    jogar qualquer um dá no mesmo, então parar no primeiro que casar é correto.
    """
    if not 0 <= target < len(hand):
        return False
    wanted = hand[target].nome
    for _ in range(max_steps):
        frame = cv2.imread(str(grab(state="seek")))
        selected = detect_card_slots(frame).selected
        if selected is None:
            return False
        current = _identify(frame, selected, db)
        if current == wanted:
            return True
        if current is None or current not in [c.nome for c in hand]:
            return False  # perdemos a referência: melhor refazer a travessia
        here = next(i for i, c in enumerate(hand) if c.nome == current)
        _tap_and_wait(selected.x, left=here > target)
    return False


def read_choices(frame: np.ndarray, db: CardDB | None = None) -> dict | None:
    """Lê as opções de uma tela de escolha recortando carta por carta.

    Mandar o frame inteiro pro modelo dava leitura ruim pela mesma razão do bug
    original de combate: a carta fica minúscula depois do resize pra 768px.
    Medido em três telas de level up reais, o caminho antigo errava o custo de 5
    das 10 cartas, devolvia descrições de um dígito ("1", "5"), inventava
    `mana=-1` e apontava a carta selecionada errada.

    None quando não há círculos de custo — aí o chamador cai no prompt de tela
    inteira, que é o caso das telas de baú sem cartas.
    """
    slots = detect_choice_slots(frame)
    if not slots.circles:
        return None
    # Escala pela MEDIANA: a carta de bônus traz um orbe decorativo no lugar do
    # círculo de custo, e usar o lado dele inflava o recorte dela em ~60%.
    side = int(median(c.side for c in slots.circles))
    opcoes = []
    for i, circle in enumerate(slots.circles):
        card = _read_choice_card(frame, circle, side, db)
        opcoes.append({
            "posicao": i,
            "nome": card.nome,
            "descricao": card.descricao,
            "mana": card.mana,
            "e_bonus": card.tipo == "bonus",
        })
    return {"opcoes": opcoes, "indice_selecionada": slots.selected_idx}


def _read_with_retry(crop: np.ndarray, db: CardDB | None) -> CardScanFrame:
    """Lê uma carta, repetindo uma vez se o custo sair ilegível.

    Custo desconhecido enfraquece a validação da jogada: `combat.validate` não
    consegue provar que a carta é cara demais, então ela passa. Uma segunda
    leitura é barata e recupera boa parte dos casos.
    """
    card = read_card(crop, db)
    return card if card.mana is not None else read_card(crop, db)


def _read_choice_card(
    frame: np.ndarray, circle: CostCircle, side: int, db: CardDB | None
) -> CardScanFrame:
    """Lê uma carta da tela de escolha, com uma segunda tentativa se o custo sair ilegível.

    Observado num recorte perfeitamente legível: o modelo devolveu `mana=None` e
    descrição truncada. É erro estocástico, não de recorte — repetir uma vez é
    barato e recupera a maioria. Não confundir com o consenso da ADR-019, que
    tirava média de leituras feitas num recorte errado.
    """
    x, y, w, h = card_bbox(circle, side)
    crop = frame[max(0, y) : y + h, max(0, x) : x + w]
    return _read_with_retry(crop, db)


_PROMPT_BY_STATE: dict[GameState, tuple[str, type[BaseModel]]] = {
    GameState.LEVEL_UP: ("level_up.txt", LevelUpState),
    GameState.SHOP: ("shop.txt", ShopState),
    GameState.CHEST: ("chest.txt", ChestState),
    GameState.CHEST_CARD_TARGET: ("chest_card_target.txt", ChestState),
    GameState.BOSS_CHEST: ("chest.txt", ChestState),
}


# Telas cujas opções são cartas com círculo de custo — leem melhor recortadas.
_CHOICE_STATES = {
    GameState.LEVEL_UP,
    GameState.CHEST,
    GameState.BOSS_CHEST,
    GameState.CHEST_CARD_TARGET,
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

    if state in _CHOICE_STATES:
        frame = cv2.imread(frame_path)
        read = read_choices(frame, default_carddb())
        if read is not None:
            return PerceivedFrame(state=state, data=read)

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
