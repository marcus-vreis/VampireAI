"""Ícones do minimapa e escolha de alvo."""

from __future__ import annotations

import collections
from pathlib import Path

import cv2
import pytest

from src.nav import plan
from src.vision.icons import IconKind, find_icons
from src.vision.minimap import Facing, Turn, locate, read_minimap

_RAIZ = Path(__file__).resolve().parent.parent
# Gabarito versionado. `frames/` é gitignored E rotacionado durante uma run,
# então teste que dependa dele passa a pular em silêncio — pior que falhar.
FRAMES = _RAIZ / "dataset" / "referencia"

# Frame conferido ícone a ícone olhando a imagem: 6 caveiras, 1 chefe, 1 "?".
REFERENCE = "20260802T154240385_combat_initial.png"
COMBAT = "20260802T154006402_map.png"  # nome enganoso: é combate


def load(name: str):
    path = FRAMES / name
    if not path.is_file():
        pytest.skip(f"frame de referência ausente: {name}")
    return cv2.imread(str(path))


def test_conta_icones_do_frame_de_referencia():
    minimap = read_minimap(load(REFERENCE))
    counts = collections.Counter(i.kind for i in find_icons(minimap.gray, minimap.arrow_side))
    assert counts[IconKind.ENEMY] == 6
    assert counts[IconKind.BOSS] == 1


def test_inimigo_morto_some_da_contagem():
    """Frame anterior da mesma run tem um inimigo a mais — o que foi derrotado."""
    antes = read_minimap(load("20260802T153406187_loop.png"))
    depois = read_minimap(load(REFERENCE))
    if antes is None:
        pytest.skip("frame anterior ausente")
    n_antes = sum(1 for i in find_icons(antes.gray, antes.arrow_side) if i.kind is IconKind.ENEMY)
    n_depois = sum(
        1 for i in find_icons(depois.gray, depois.arrow_side) if i.kind is IconKind.ENEMY
    )
    assert n_antes == n_depois + 1


def test_nao_acha_minimapa_em_combate():
    """A arte das cartas cai na mesma faixa de cinza do pergaminho.

    Sem exigir densidade de pergaminho, a busca devolvia a área toda e um círculo
    de custo azul virava seta do jogador.
    """
    assert locate(load(COMBAT)) is None
    assert read_minimap(load(COMBAT)) is None


def test_icones_nao_furam_a_area_andavel():
    """Caveira é cinza 136, abaixo do limiar de piso — precisa ser andável.

    Senão o BFS não chega no inimigo, que é justamente o alvo.
    """
    minimap = read_minimap(load(REFERENCE))
    for icon in find_icons(minimap.gray, minimap.arrow_side):
        if icon.kind is IconKind.QUESTION:
            continue
        assert minimap.walkable[icon.y, icon.x], f"{icon.kind.value} em ({icon.x},{icon.y})"


def test_plano_mira_inimigo_antes_de_explorar():
    step = plan(read_minimap(load(REFERENCE)))
    assert step is not None
    assert step.reason == "inimigo mais próximo"


@pytest.mark.parametrize(
    ("frame", "facing", "turn"),
    [
        ("20260802T153406187_loop.png", Facing.EAST, Turn.FORWARD),
        ("20260802T153746341_loop.png", Facing.NORTH, Turn.RIGHT),
        ("20260802T153756843_loop.png", Facing.WEST, Turn.BACK),
    ],
)
def test_mesma_posicao_direcoes_diferentes_apontam_pro_mesmo_lugar(frame, facing, turn):
    """Da mesma casa, olhando pra três lados, o giro tem que convergir no leste.

    É a checagem mais forte que dá pra fazer sem o jogo rodando: se o facing ou a
    aritmética de giro estivessem errados, as três respostas não seriam coerentes.
    """
    minimap = read_minimap(load(frame))
    assert minimap.facing is facing
    step = plan(minimap)
    assert step is not None
    assert step.turn is turn


def sem_icones(monkeypatch):
    import src.nav

    monkeypatch.setattr(src.nav, "find_icons", lambda *a, **k: [])


def test_cascata_de_prioridade_desce_ao_derrotar_alvos(monkeypatch):
    """Simulação de fim de fase: apaga os ícones do minimapa e vê o plano descer.

    Não há frame de fim de fase no repositório, mas o comportamento é testável
    removendo os ícones do mapa real.
    """
    import dataclasses

    import numpy as np

    from src.vision.minimap import read_minimap

    minimap = read_minimap(load(REFERENCE))
    assert plan(minimap).reason == "inimigo mais próximo"

    sem_icone = dataclasses.replace(minimap)
    sem_icones(monkeypatch)
    assert plan(sem_icone).reason == "explorar fronteira"

    revelado = dataclasses.replace(minimap, fog=np.zeros_like(minimap.fog))
    assert plan(revelado).reason == "procurar a saída da fase"


def test_alvos_ficam_na_componente_do_jogador():
    """O minimapa mostra salas ainda sem corredor aberto: 15 componentes conexas
    no frame de referência. Alvo em outra é visível e inalcançável, e o BFS
    gastava a busca inteira pra devolver None."""
    from src.vision.minimap import direction_to, distant_floor, read_minimap

    minimap = read_minimap(load(REFERENCE))
    alvos = distant_floor(minimap)
    assert alvos
    assert all(direction_to(minimap, alvo) is not None for alvo in alvos)


def test_fim_de_fase_nao_devolve_none(monkeypatch):
    """Antes, `plan` devolvia None e o handler andava pra frente às cegas até o
    antitravamento abortar — justamente quando havia uma fase pra avançar."""
    import dataclasses

    import numpy as np

    from src.vision.minimap import read_minimap

    sem_icones(monkeypatch)
    minimap = read_minimap(load(REFERENCE))
    fim = dataclasses.replace(minimap, fog=np.zeros_like(minimap.fog))
    assert plan(fim) is not None


def test_bonus_ficam_colados_na_borda_da_sala():
    """A observação que motivou o recurso: itens não ficam no centro do bloco.

    `jogo.md`: "as vezes eles ficam colado a paredes, então não são só blocos".
    Medido: todos a 0-3px da borda do piso.
    """
    import cv2
    import numpy as np

    from src.vision.minimap import find_bonuses, read_minimap

    minimap = read_minimap(load(REFERENCE))
    bonuses = find_bonuses(minimap)
    assert bonuses, "o frame de referência tem pontos de bônus visíveis"

    piso = minimap.walkable.astype(np.uint8)
    borda = cv2.dilate(piso, np.ones((3, 3), np.uint8)) - cv2.erode(piso, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(borda)
    for bx, by in bonuses:
        dist = np.sqrt(((xs - bx) ** 2 + (ys - by) ** 2).min())
        assert dist <= 4, f"bônus em ({bx},{by}) está a {dist:.0f}px da borda"


def test_bonus_estao_sobre_o_piso():
    """Textura do pergaminho fora da sala cai no mesmo tom e precisa ser excluída."""
    from src.vision.minimap import find_bonuses, read_minimap

    minimap = read_minimap(load(REFERENCE))
    for bx, by in find_bonuses(minimap):
        assert minimap.walkable[by, bx]


def test_bonus_nao_sao_confundidos_com_inimigos():
    """Caveira mede 13px; bônus, 4-6px. Os filtros de tamanho não se sobrepõem."""
    from src.vision.icons import find_icons
    from src.vision.minimap import find_bonuses, read_minimap

    minimap = read_minimap(load(REFERENCE))
    icones = {(i.x, i.y) for i in find_icons(minimap.gray, minimap.arrow_side)}
    for bx, by in find_bonuses(minimap):
        assert all(abs(bx - ix) > 8 or abs(by - iy) > 8 for ix, iy in icones)


def test_bonus_nao_viram_alvo_de_navegacao():
    """jogo.md classifica bônus como "totalmente ignorável, apenas pegue caso
    estejam no caminho". Pegá-los exige virar pra parede e andar, manobra que a
    navegação por células não faz — e agir com falso positivo faria o agente
    andar contra parede até o antitravamento disparar."""
    minimap = read_minimap(load(REFERENCE))
    step = plan(minimap)
    assert step is not None
    assert "bônus" not in step.reason


# --- baus no minimapa ------------------------------------------------------
# Recortados do frame 20260901T193925141 (fase 2/4), o primeiro do dataset a
# mostrar os dois tipos. Os 8 mapas da fase 1/4 nao tem bau nenhum e servem de
# teste negativo -- e o unico jeito honesto de medir falso positivo num
# template que e basicamente um retangulo cheio.
_MAPAS_FASE1 = (
    "20260830T134955063_label_map.png",
    "20260830T170626089_label_map.png",
    "20260901T193104652_label_map.png",
    "20260901T193203009_label_map.png",
    "20260901T193230959_label_map.png",
    "20260901T193606851_label_map.png",
    "20260901T193704035_label_map.png",
    "20260901T193755377_label_map.png",
)
_DATASET = Path(__file__).resolve().parent.parent / "dataset"


def _icones(nome):
    import collections

    from src.vision.icons import find_icons
    from src.vision.minimap import read_minimap

    caminho = _DATASET / nome
    if not caminho.is_file():
        pytest.skip(f"frame ausente: {nome}")
    mm = read_minimap(cv2.imread(str(caminho)))
    assert mm is not None, "minimapa nao localizado"
    return collections.Counter(i.kind.value for i in find_icons(mm.gray, mm.arrow_side))


def test_acha_os_dois_tipos_de_bau():
    c = _icones("20260901T193925141_label_map.png")
    assert c["bau"] == 1, "bau comum: tampa reta com trinco, na borda esquerda"
    assert c["bau_chefe"] == 1, "bau ornamentado: tampa arredondada com X nas laterais"


def test_o_bau_some_do_mapa_quando_e_recolhido():
    """Confirmacao independente do detector: os dois frames da fase 2 sao da
    mesma run, e entre eles o jogador pegou o bau comum. Os blobs em (56,138)
    existem so no primeiro. Um detector que "achasse" bau nos dois estaria
    casando com parede, nao com bau."""
    c = _icones("20260901T194024213_label_map.png")
    assert c["bau"] == 0
    assert c["bau_chefe"] == 1, "o ornamentado continua no mapa"


@pytest.mark.parametrize("nome", _MAPAS_FASE1)
def test_nenhum_falso_positivo_de_bau_na_fase_1(nome):
    c = _icones(nome)
    assert c["bau"] == 0 and c["bau_chefe"] == 0


def test_os_baus_entram_na_area_andavel():
    """Os icones sao cinza 136, ABAIXO do limiar de piso. Sem soma-los a area
    andavel eles viram buracos e o BFS nao alcanca a celula do bau."""
    from src.vision.minimap import _ICON_HI, _ICON_LO

    tpl = cv2.imread(str(Path("src/vision/templates/chest.png")), cv2.IMREAD_GRAYSCALE)
    assert tpl is not None
    assert (tpl >= _ICON_LO).any() and (tpl <= _ICON_HI).any(), (
        "o tom do bau precisa cair na faixa que _icons_on_floor recupera"
    )
