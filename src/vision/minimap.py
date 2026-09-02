"""Leitura do minimapa: posição, direção e caminho até um alvo.

Substitui a pergunta "o alvo está à frente, esquerda, direita ou atrás?" que era
feita ao VLM a partir da visão em 1ª pessoa — raciocínio espacial 3D é das coisas
mais difíceis pra um modelo pequeno, e a resposta já está desenhada no minimapa
ao lado.

Não assumimos tamanho de célula: o minimapa muda de zoom entre fases (a seta do
jogador mede 16px numa e 19px noutra). O caminho é buscado em espaço de pixels
sobre a máscara de área andável, e só a DIREÇÃO do primeiro trecho é usada. Como
cada passo recaptura, não há erro acumulado.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

_MINIMAP_BOX = (705, 512, 1010, 720)
# Margem de busca em volta da caixa nominal, pra localizar o pergaminho por âncora.
_SEARCH_BOX = (640, 440, 1180, 720)
_PARCHMENT_LO, _PARCHMENT_HI = 140, 235
_MIN_PARCHMENT_AREA = 12_000
# Mapa de verdade fica em ~0.88; combate (arte de carta na mesma faixa), ~0.17.
_MIN_PARCHMENT_RATIO = 0.50

# A seta do jogador é o único azul saturado sobre o pergaminho.
_ARROW_LO, _ARROW_HI = np.array([100, 150, 80]), np.array([125, 255, 255])
_ARROW_MIN_AREA, _ARROW_MAX_AREA = 40, 400
_ARROW_MIN_SIDE, _ARROW_MAX_SIDE = 8, 26

# Três níveis no pergaminho: piso conhecido (claro), névoa (tan médio) e vazio
# fora do mapa (escuro). A fronteira de exploração é piso encostado em névoa.
_WALKABLE_MIN = 185
_FOG_MIN = 140
# Caveiras, chefe, bônus e "?" são desenhados neste tom único.
_ICON_LO, _ICON_HI = 128, 146
_SEARCH_STRIDE = 3  # BFS num subamostrado: 3px basta e é ~10x mais rápido


class Facing(str, Enum):
    NORTH = "norte"
    EAST = "leste"
    SOUTH = "sul"
    WEST = "oeste"


class Turn(str, Enum):
    FORWARD = "frente"
    LEFT = "esquerda"
    RIGHT = "direita"
    BACK = "atras"


_DELTA: dict[Facing, tuple[int, int]] = {
    Facing.NORTH: (0, -1),
    Facing.EAST: (1, 0),
    Facing.SOUTH: (0, 1),
    Facing.WEST: (-1, 0),
}
_ORDER = [Facing.NORTH, Facing.EAST, Facing.SOUTH, Facing.WEST]


@dataclass(frozen=True)
class Minimap:
    player: tuple[int, int]  # centro da seta, em pixels do minimapa
    facing: Facing
    walkable: np.ndarray  # piso conhecido
    fog: np.ndarray  # área ainda não revelada
    arrow_side: int  # lado da seta em px — proxy de escala do zoom atual
    box: tuple[int, int, int, int]  # onde o minimapa foi achado no frame
    gray: np.ndarray  # recorte em escala de cinza, pra busca de ícones


def _arrow(hsv: np.ndarray) -> tuple[int, int, int, int, np.ndarray] | None:
    mask = cv2.inRange(hsv, _ARROW_LO, _ARROW_HI)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if not (_ARROW_MIN_AREA < area < _ARROW_MAX_AREA):
            continue
        if not (_ARROW_MIN_SIDE <= w <= _ARROW_MAX_SIDE and _ARROW_MIN_SIDE <= h <= _ARROW_MAX_SIDE):
            continue
        return x, y, w, h, labels == i
    return None


def _facing_from_mass(x: int, y: int, w: int, h: int, blob: np.ndarray) -> Facing:
    """Direção pelo deslocamento do centroide.

    A seta é um triângulo: a base concentra massa do lado oposto à ponta, então o
    centroide cai atrás do centro do bbox. O vetor centro→centroide invertido
    aponta pra onde o jogador olha.
    """
    ys, xs = np.nonzero(blob)
    dx = (x + w / 2) - xs.mean()
    dy = (y + h / 2) - ys.mean()
    if abs(dx) >= abs(dy):
        return Facing.EAST if dx > 0 else Facing.WEST
    return Facing.SOUTH if dy > 0 else Facing.NORTH


def _icons_on_floor(gray: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """Pixels de ícone que estão sobre o piso.

    Caveiras, chefe e bônus são desenhados em cinza 136, abaixo do limiar de piso
    — sem isto eles viram buracos na máscara e o BFS não consegue chegar até um
    inimigo, que é exatamente o alvo. As bordas rasgadas do pergaminho usam o
    mesmo tom, por isso a exigência de estar perto do piso.
    """
    tone = cv2.inRange(gray, _ICON_LO, _ICON_HI)
    near_floor = cv2.dilate(floor.astype(np.uint8), np.ones((9, 9), np.uint8))
    return (tone & (near_floor * 255)).astype(bool)


def _parchment_fraction(frame: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = box
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float(((gray > _PARCHMENT_LO) & (gray < _PARCHMENT_HI)).mean())


def locate(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Caixa do minimapa, achada pelo bloco de pergaminho. None se não há mapa.

    A caixa fixa quebrava quando a captura saía desalinhada — há frames em que ela
    pega o viewport 3D no topo e perde a metade de baixo do mapa. Ancorar na maior
    região cor-de-pergaminho é barato e sobrevive a isso.

    A caixa precisa PROVAR que é pergaminho: sem essa checagem, num frame de
    combate a busca devolvia a área inteira (arte de carta cai na mesma faixa de
    cinza) e um círculo de custo azul passava por seta do jogador. Mapa de verdade
    fica em ~0.88 de pergaminho; combate, em ~0.17.
    """
    sx0, sy0, sx1, sy1 = _SEARCH_BOX
    gray = cv2.cvtColor(frame[sy0:sy1, sx0:sx1], cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, _PARCHMENT_LO, _PARCHMENT_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best = max(range(1, n), key=lambda i: stats[i][4], default=None)
    if best is None or stats[best][4] < _MIN_PARCHMENT_AREA:
        return None
    x, y, w, h, _ = (int(v) for v in stats[best])
    box = (sx0 + x, sy0 + y, sx0 + x + w, sy0 + y + h)
    if _parchment_fraction(frame, box) < _MIN_PARCHMENT_RATIO:
        return None
    return box


def read_minimap(frame: np.ndarray) -> Minimap | None:
    """Lê o minimapa de um frame BGR. None se a seta do jogador não aparece."""
    box = locate(frame)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    mm = frame[y0:y1, x0:x1]
    found = _arrow(cv2.cvtColor(mm, cv2.COLOR_BGR2HSV))
    if found is None:
        return None
    ax, ay, aw, ah, blob = found
    gray = cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY)
    floor = gray > _WALKABLE_MIN
    walkable = floor | _icons_on_floor(gray, floor)
    walkable[blob] = True  # a própria seta cobre o piso onde o jogador está
    return Minimap(
        player=(ax + aw // 2, ay + ah // 2),
        facing=_facing_from_mass(ax, ay, aw, ah, blob),
        walkable=walkable,
        fog=(gray > _FOG_MIN) & ~walkable,
        arrow_side=max(aw, ah),
        box=(x0, y0, x1, y1),
        gray=gray,
    )


def _bfs_step(mm: Minimap, target: tuple[int, int]) -> tuple[int, int] | None:
    """Primeiro passo do caminho mais curto até `target`, em pixels do minimapa.

    Roda num grid subamostrado por `_SEARCH_STRIDE` — a resolução do minimapa é
    muito maior que a precisão necessária pra escolher uma direção.
    """
    s = _SEARCH_STRIDE
    h, w = mm.walkable.shape
    small = mm.walkable[::s, ::s]
    start = (mm.player[0] // s, mm.player[1] // s)
    goal = (target[0] // s, target[1] // s)
    sh, sw = small.shape
    if not (0 <= goal[0] < sw and 0 <= goal[1] < sh):
        return None

    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        cx, cy = cur
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nxt = (cx + dx, cy + dy)
            if nxt in prev:
                continue
            if not (0 <= nxt[0] < sw and 0 <= nxt[1] < sh):
                continue
            if not small[nxt[1], nxt[0]]:
                continue
            prev[nxt] = cur
            q.append(nxt)
    if goal not in prev:
        return None

    node = goal
    while prev[node] is not None and prev[node] != start:
        node = prev[node]
    if prev[node] is None:
        return None
    return (node[0] * s, node[1] * s)


# Lado da célula do minimapa. Medido nos 10 mapas do `dataset/`, fases 1/4 e 2/4:
# a moda dos trechos andáveis é 19px em TODOS, mesmo com a seta do jogador
# variando entre 15 e 16px — o zoom muda o desenho da seta, não o grid.
_CELL_PX = 19

_STEP = {
    Facing.NORTH: (0, -1),
    Facing.SOUTH: (0, 1),
    Facing.EAST: (1, 0),
    Facing.WEST: (-1, 0),
}


def cell_ahead_walkable(mm: Minimap, facing: Facing) -> bool:
    """A célula inteira à frente é piso?

    `_bfs_step` responde num grid de `_SEARCH_STRIDE` (3px), um SEXTO da célula.
    Perguntar a ele "dá pra andar pra frente?" é perguntar sobre um ponto DENTRO
    da célula onde o jogador já está — que obviamente é piso. Foi assim que duas
    runs reais mandaram andar contra parede até o antitravamento abortar: em
    (63,39) olhando leste e em (124,39) olhando sul, com a faixa andável medindo
    19px e o jogador a 9px da parede. Ver ADR-087.
    """
    dx, dy = _STEP[facing]
    px, py = mm.player
    x, y = px + dx * _CELL_PX, py + dy * _CELL_PX
    h, w = mm.walkable.shape
    if not (0 <= x < w and 0 <= y < h):
        return False
    return bool(mm.walkable[y, x])


def direction_to(mm: Minimap, target: tuple[int, int]) -> Turn | None:
    """Para onde virar/andar pra avançar rumo ao alvo. None se não há caminho.

    Girar é sempre seguro, então a checagem de parede só se aplica quando a
    resposta seria ANDAR: aí a célula à frente precisa ser piso de verdade.
    """
    step = _bfs_step(mm, target)
    if step is None:
        return None
    dx, dy = step[0] - mm.player[0], step[1] - mm.player[1]
    if dx == 0 and dy == 0:
        want = mm.facing
    else:
        want = (
            (Facing.EAST if dx > 0 else Facing.WEST)
            if abs(dx) >= abs(dy)
            else (Facing.SOUTH if dy > 0 else Facing.NORTH)
        )
    if not cell_ahead_walkable(mm, want):
        return None
    return relative_turn(mm.facing, want)


def relative_turn(current: Facing, want: Facing) -> Turn:
    """Quantos giros separam a direção atual da desejada."""
    diff = (_ORDER.index(want) - _ORDER.index(current)) % 4
    return [Turn.FORWARD, Turn.RIGHT, Turn.BACK, Turn.LEFT][diff]


# Bônus no minimapa: manchas pequenas no tom de ícone, sobre o piso e coladas na
# borda da sala. Medido no frame de referência: 4-6px de lado, a 0-3px da borda.
_BONUS_MIN_SIDE, _BONUS_MAX_SIDE = 3, 12
_BONUS_MIN_AREA = 12
_BONUS_MAX_EDGE_DIST = 4


def find_bonuses(mm: Minimap) -> list[tuple[int, int]]:
    """Pontos de bônus, que o jogo desenha COLADOS na parede do bloco.

    Não ficam no centro da célula: `jogo.md` descreve que "as vezes eles ficam
    colado a paredes", e a medição confirma — todos a 0-3px da borda da sala.
    Pegá-los exige virar pra parede e andar, manobra que a navegação por células
    não faz.

    Devolve as posições; **nada age sobre elas ainda**. `jogo.md` classifica bônus
    como "totalmente ignorável, apenas pegue caso estejam no caminho", e a taxa
    de falso positivo (textura do pergaminho no mesmo tom) ainda não foi medida
    contra um conjunto rotulado. Expor no overlay de debug é o passo que torna
    isso mensurável.
    """
    tone = cv2.inRange(mm.gray, _ICON_LO, _ICON_HI)
    floor = mm.walkable.astype(np.uint8)
    near_floor = cv2.dilate(floor, np.ones((5, 5), np.uint8))
    edge = cv2.dilate(floor, np.ones((3, 3), np.uint8)) - cv2.erode(
        floor, np.ones((3, 3), np.uint8)
    )
    ys_edge, xs_edge = np.nonzero(edge)
    if len(xs_edge) == 0:
        return []

    total, _, stats, centroids = cv2.connectedComponentsWithStats(
        tone & (near_floor * 255), 8
    )
    found: list[tuple[int, int]] = []
    for i in range(1, total):
        x, y, w, h, area = (int(v) for v in stats[i])
        if not (_BONUS_MIN_SIDE <= w <= _BONUS_MAX_SIDE and _BONUS_MIN_SIDE <= h <= _BONUS_MAX_SIDE):
            continue
        if area < _BONUS_MIN_AREA:
            continue
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        if not mm.walkable[cy, cx]:
            continue  # textura do pergaminho fora da sala
        dist = np.sqrt(((xs_edge - cx) ** 2 + (ys_edge - cy) ** 2).min())
        if dist <= _BONUS_MAX_EDGE_DIST:
            found.append((cx, cy))
    return found


def reachable(mm: Minimap) -> np.ndarray:
    """Piso ligado ao jogador. O minimapa mostra salas ainda sem corredor aberto.

    Medido no frame de referência: o piso tem 15 componentes conexas, e a do
    jogador é uma. Alvos nas outras são visíveis e inalcançáveis — o BFS gasta a
    busca inteira e devolve None.
    """
    total, labels = cv2.connectedComponents(mm.walkable.astype(np.uint8), 4)
    if total <= 1:
        return mm.walkable
    px, py = mm.player
    return labels == labels[py, px]


def distant_floor(mm: Minimap) -> list[tuple[int, int]]:
    """Piso conhecido, do mais DISTANTE do jogador pro mais próximo.

    Último recurso quando não há inimigo, chefe nem névoa: o mapa está revelado
    mas a saída da fase ainda precisa ser alcançada. Andar pro ponto mais longe
    move o agente por regiões que ele não percorreu, o que é estritamente melhor
    que a alternativa anterior — andar pra frente às cegas até o antitravamento
    abortar a run bem na hora de avançar de fase.
    """
    ys, xs = np.nonzero(reachable(mm))
    if len(xs) == 0:
        return []
    px, py = mm.player
    ordem = np.argsort(-((xs - px) ** 2 + (ys - py) ** 2))
    return [(int(xs[i]), int(ys[i])) for i in ordem[::11][:32]]


def frontier_targets(mm: Minimap) -> list[tuple[int, int]]:
    """Piso conhecido encostado na névoa, do mais próximo ao mais distante.

    Alvos precisam ser andáveis, senão o BFS não os alcança — por isso é a borda
    INTERNA do piso, não a externa. Encostar em névoa (e não em vazio) evita
    mandar o agente pra parede sem saída.

    Sem classificar ícones ainda: explorar a fronteira já faz o agente percorrer
    a fase. Priorizar caveira/chefe entra quando houver dataset rotulado.
    """
    walk = reachable(mm).astype(np.uint8)
    near_fog = cv2.dilate(mm.fog.astype(np.uint8), np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero(walk & near_fog)
    if len(xs) == 0:
        return []
    px, py = mm.player
    order = np.argsort((xs - px) ** 2 + (ys - py) ** 2)
    return [(int(xs[i]), int(ys[i])) for i in order[::7][:48]]
