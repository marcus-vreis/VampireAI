"""Ícones do minimapa e escolha de alvo."""

from __future__ import annotations

import collections
from pathlib import Path

import cv2
import pytest

from src.nav import plan
from src.vision.icons import IconKind, find_icons
from src.vision.minimap import Facing, Turn, locate, read_minimap

FRAMES = Path(__file__).resolve().parent.parent / "frames"

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
    counts = collections.Counter(
        i.kind for i in find_icons(minimap.gray, minimap.arrow_side)
    )
    assert counts[IconKind.ENEMY] == 6
    assert counts[IconKind.BOSS] == 1


def test_inimigo_morto_some_da_contagem():
    """Frame anterior da mesma run tem um inimigo a mais — o que foi derrotado."""
    antes = read_minimap(load("20260802T153406187_loop.png"))
    depois = read_minimap(load(REFERENCE))
    if antes is None:
        pytest.skip("frame anterior ausente")
    n_antes = sum(1 for i in find_icons(antes.gray, antes.arrow_side) if i.kind is IconKind.ENEMY)
    n_depois = sum(1 for i in find_icons(depois.gray, depois.arrow_side) if i.kind is IconKind.ENEMY)
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
