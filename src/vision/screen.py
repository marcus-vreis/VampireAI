"""Classificação de tela por assinatura de pixels.

Substitui a chamada de VLM que era a mais frequente do agente (95 de 319 no log)
e a menos confiável: dos 39 frames que são o mapa, o modelo rotulou ao menos 9
como outra coisa — chegando a rodar um scan de cartas de 4 passos em cima do
mapa. As telas do jogo têm assinaturas de cor bem separadas; medir é melhor que
perguntar.

O classificador não tenta cobrir tudo. Ele resolve com certeza os casos que
dominam o loop (mapa, combate) e delega os raros ao VLM — com a lista de opções
já restrita, o que também torna a pergunta mais fácil pra ele.

A tela "Baralho" entrou depois, achada observando o jogo ao vivo: as cartas do
deck também têm círculo de custo, então ela passava por combate e o agente
tentaria jogar carta ali.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from src.vision.cards import detect_card_slots

# Caixas em coordenadas do client area 1280x720.
_MINIMAP_BOX = (705, 512, 1010, 720)
_CENTER_BOX = (260, 120, 1030, 580)
_HEART_BOX = (150, 455, 265, 570)
_ORB_BOX = (1015, 455, 1145, 585)
_PLAY_ALL_BOX = (165, 590, 245, 630)
_END_TURN_BOX = (1032, 590, 1114, 630)
_CHEST_BUTTON_BOX = (470, 612, 808, 658)
_GAME_OVER_BUTTON_BOX = (565, 495, 715, 550)

# Limiares. As margens são largas: pergaminho do minimapa fica em 0.62-0.76 no
# mapa e no máximo 0.32 em combate; o painel slate de diálogo, 0.56-0.63 contra
# no máximo 0.11 em qualquer outra tela.
_PARCHMENT_MAP = 0.50
_SLATE_DIALOG = 0.45
# A tela "Baralho" mostra o deck inteiro num painel de tamanho intermediário.
# Medido: combate fica em 0.017-0.052, baralho em 0.181, diálogo em 0.56-0.63.
# O limiar sozinho não bastava: o menu principal fica em 0.104 e era classificado
# como baralho. O que separa não é ajuste de número, é uma regra do jogo — só dá
# pra abrir o baralho DENTRO de uma run, então o HUD (coração e orbe) tem que
# estar presente. Ver ADR-049.
_SLATE_DECK = 0.10
_HUD_PRESENT = 0.02
# Botões bege "Jogar todas" e "Finalizar turno": presentes em 100% dos frames de
# combate (0.734-0.997, inclusive com mão vazia entre turnos) e ausentes (0.000)
# em mapa, baú, game over e baralho.
_TURN_BUTTON_BRIGHT = 110
_TURN_BUTTON_PRESENT = 0.50
# Telas de baú (chest, boss_chest, chest_card_target) não têm painel slate
# (0.015-0.017), mas exibem o botão vermelho inferior ("Sair"/"Sacar") em 0.801
# contra no máximo 0.195 em qualquer outra tela do dataset.
_CHEST_BUTTON_PRESENT = 0.50
# Tela de Game Over mantém o HUD visível mas exibe o botão azul "Pronto" no
# centro inferior (0.601 contra <= 0.05 em combate/mapa).
_GAME_OVER_BUTTON_PRESENT = 0.40


class Verdict(str, Enum):
    """Resultado da classificação por CV."""

    MAP = "map"
    COMBAT = "combat"
    DECK = "deck"  # tela "Baralho", o deck inteiro
    DIALOG = "dialog"  # level_up / chest / chest_card_target / boss_chest
    UNKNOWN = "unknown"  # delega ao VLM
    NOT_GAME = "not_game"  # a captura não pegou o jogo


@dataclass(frozen=True)
class ScreenSignature:
    parchment: float
    slate: float
    cards: int
    hud: bool
    brightness: float
    verdict: Verdict


def _box(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return frame[y0:y1, x0:x1]


def _parchment_ratio(frame: np.ndarray) -> float:
    """Fração do minimapa com a cor de pergaminho. Alto só quando o mapa é visível."""
    gray = cv2.cvtColor(_box(frame, _MINIMAP_BOX), cv2.COLOR_BGR2GRAY)
    return float(((gray > 150) & (gray < 235)).mean())


def _slate_ratio(frame: np.ndarray) -> float:
    """Fração do centro ocupada pelo painel azul-ardósia dos diálogos."""
    hsv = cv2.cvtColor(_box(frame, _CENTER_BOX), cv2.COLOR_BGR2HSV)
    slate = cv2.inRange(hsv, np.array([100, 20, 60]), np.array([135, 110, 190]))
    return float((slate > 0).mean())


def _hud_present(frame: np.ndarray) -> bool:
    """Coração de HP e orbe de mana visíveis — marca de que uma run está em curso."""
    heart = cv2.cvtColor(_box(frame, _HEART_BOX), cv2.COLOR_BGR2HSV)
    red = cv2.inRange(heart, np.array([0, 120, 70]), np.array([10, 255, 255])) | cv2.inRange(
        heart, np.array([170, 120, 70]), np.array([180, 255, 255])
    )
    orb = cv2.cvtColor(_box(frame, _ORB_BOX), cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(orb, np.array([95, 120, 70]), np.array([125, 255, 255]))
    return float((red > 0).mean()) > _HUD_PRESENT and float((blue > 0).mean()) > _HUD_PRESENT


def _turn_buttons_present(frame: np.ndarray) -> bool:
    """Placas bege 'Jogar todas' e 'Finalizar turno' visíveis no HUD de combate."""
    left = cv2.cvtColor(_box(frame, _PLAY_ALL_BOX), cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(_box(frame, _END_TURN_BOX), cv2.COLOR_BGR2GRAY)
    ratio = min(
        float((left > _TURN_BUTTON_BRIGHT).mean()), float((right > _TURN_BUTTON_BRIGHT).mean())
    )
    return ratio >= _TURN_BUTTON_PRESENT


def _chest_button_present(frame: np.ndarray) -> bool:
    """Botão vermelho inferior ('Sair'/'Sacar') das telas de baú e alvo de joia."""
    hsv = cv2.cvtColor(_box(frame, _CHEST_BUTTON_BOX), cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, np.array([0, 120, 80]), np.array([12, 255, 255])) | cv2.inRange(
        hsv, np.array([170, 120, 80]), np.array([180, 255, 255])
    )
    return float((red > 0).mean()) >= _CHEST_BUTTON_PRESENT


def _game_over_button_present(frame: np.ndarray) -> bool:
    """Botão azul 'Pronto' centralizado abaixo do banner de Game Over."""
    hsv = cv2.cvtColor(_box(frame, _GAME_OVER_BUTTON_BOX), cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([105, 130, 120]), np.array([128, 255, 255]))
    return float((blue > 0).mean()) >= _GAME_OVER_BUTTON_PRESENT


def signature(frame: np.ndarray) -> ScreenSignature:
    """Mede a tela e emite um veredito. Não faz nenhuma chamada de modelo."""
    parchment = _parchment_ratio(frame)
    slate = _slate_ratio(frame)
    cards = detect_card_slots(frame).visible_total
    hud = _hud_present(frame)
    turn_buttons = _turn_buttons_present(frame)
    chest_button = _chest_button_present(frame)
    game_over_button = _game_over_button_present(frame)
    brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())

    return ScreenSignature(
        parchment=round(parchment, 3),
        slate=round(slate, 3),
        cards=cards,
        hud=hud,
        brightness=round(brightness, 1),
        verdict=_verdict(
            parchment, slate, cards, hud, turn_buttons, chest_button, game_over_button
        ),
    )


def _verdict(
    parchment: float,
    slate: float,
    cards: int,
    hud: bool,
    turn_buttons: bool = False,
    chest_button: bool = False,
    game_over_button: bool = False,
) -> Verdict:
    if slate >= _SLATE_DIALOG or (chest_button and hud):
        return Verdict.DIALOG
    if slate >= _SLATE_DECK and hud:
        # As cartas do deck também têm círculo de custo, então sem esta checagem
        # a tela "Baralho" passava por combate e o agente tentava jogar carta ali.
        # O HUD é o que a separa do menu principal, que tem painel de tamanho
        # parecido mas acontece FORA de uma run.
        return Verdict.DECK
    if parchment >= _PARCHMENT_MAP:
        return Verdict.MAP
    if turn_buttons or cards >= 2:
        return Verdict.COMBAT
    if hud and not game_over_button:
        # HUD sem placas de turno e sem mão: tela de escolha por cima da run. Era
        # combate, e "Inserir joia" com UMA carta virava laço de baú (ADR-101).
        # Placas presentes em 12 de 13 combates rotulados; o 13º já sai como mapa.
        return Verdict.DIALOG
    if not hud and parchment < 0.1 and slate < 0.1:
        return Verdict.NOT_GAME
    return Verdict.UNKNOWN
