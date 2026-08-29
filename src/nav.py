"""Escolha de alvo no mapa e tradução em micro-ação de controle.

Compõe `vision.minimap` (onde dá pra andar) com `vision.icons` (o que há pra
visitar). Antes, "pra onde vou?" era pergunta ao VLM sobre a visão em 1ª pessoa;
agora é busca em grafo sobre o mapa que o jogo já desenha.

Prioridade, segundo `jogo.md`: limpar os inimigos menores deixa o personagem
forte pro chefe, então inimigo vem antes de chefe. Bônus só se estiver no
caminho — não vale desviar.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.vision.icons import Icon, IconKind, find_icons
from src.vision.minimap import Minimap, Turn, direction_to, frontier_targets


@dataclass(frozen=True)
class Plan:
    turn: Turn
    goal: tuple[int, int]
    reason: str


def _distance2(minimap: Minimap, point: tuple[int, int]) -> int:
    px, py = minimap.player
    return (point[0] - px) ** 2 + (point[1] - py) ** 2


def _nearest_first(minimap: Minimap, icons: list[Icon], kind: IconKind) -> list[tuple[int, int]]:
    pts = [(i.x, i.y) for i in icons if i.kind is kind]
    return sorted(pts, key=lambda p: _distance2(minimap, p))


def _first_reachable(minimap: Minimap, goals: list[tuple[int, int]], reason: str) -> Plan | None:
    for goal in goals:
        turn = direction_to(minimap, goal)
        if turn is not None:
            return Plan(turn=turn, goal=goal, reason=reason)
    return None


def plan(minimap: Minimap) -> Plan | None:
    """Próxima micro-ação no mapa, ou None se nada é alcançável.

    O jogador só enxerga ícones em área já revelada, então explorar a fronteira
    continua sendo o plano de fundo — é o que faz novos alvos aparecerem.
    """
    icons = find_icons(minimap.gray, minimap.arrow_side)
    candidates = [
        (_nearest_first(minimap, icons, IconKind.ENEMY), "inimigo mais próximo"),
        (_nearest_first(minimap, icons, IconKind.BOSS), "chefe"),
        (frontier_targets(minimap), "explorar fronteira"),
    ]
    for goals, reason in candidates:
        found = _first_reachable(minimap, goals, reason)
        if found is not None:
            return found
    return None
