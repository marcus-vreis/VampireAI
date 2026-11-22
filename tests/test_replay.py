"""Teste de integração: a máquina de estados inteira sobre frames salvos.

Todo o resto da suíte é unitário. Este é o único que exercita percepção →
roteamento → handler → ação no mesmo caminho que o agente usa, e existe porque um
bug real escapou dos unitários: o HP era lido, `_hp_line` passava no teste, e o
dado não chegava ao prompt.

Roda em modo offline (nenhuma chamada de modelo) e com o gamepad em dry-run, então
é rápido e não toca em nada de fora.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import gamepad
from src.replay import replay

_RAIZ = Path(__file__).resolve().parent.parent
# Gabarito versionado. `frames/` é gitignored E rotacionado durante uma run,
# então teste que dependa dele passa a pular em silêncio — pior que falhar.
FRAMES = _RAIZ / "dataset" / "referencia"


@pytest.fixture(scope="module")
def report():
    if not FRAMES.is_dir() or not any(FRAMES.glob("*.png")):
        pytest.skip("sem frames de referência")
    return replay(FRAMES, offline=True)


def test_gamepad_fica_em_dry_run(report):
    """A garantia mais importante: o replay não pode emitir input de verdade."""
    assert gamepad.is_dry_run()


def test_nenhum_handler_estoura(report):
    assert report.errors == [], "\n".join(report.errors)


def test_processa_todos_os_frames(report):
    """O conjunto de referência é curado e versionado: 18 frames conferidos um a
    um. Antes isto rodava sobre `frames/`, cujo conteúdo variava a cada sessão —
    o teste media coisa diferente a cada execução."""
    assert report.frames == len(list(FRAMES.glob("*.png")))


def test_reconhece_o_mapa_sem_modelo(report):
    """Se isto caísse pra zero, a CV teria parado de resolver o caminho comum.

    Combate não aparece aqui: os frames de combate do conjunto de referência têm
    cartas, então o handler chama o modelo pra lê-las e o passo conta como
    "exigiu modelo" em vez de virar estado.
    """
    assert report.states["map"] >= 5


def test_captura_fora_do_jogo_vira_estado_proprio(report):
    """Há frames em que a captura pegou outra janela. Chutar um estado ali é pior."""
    assert report.states["not_game"] >= 1


def test_reconhece_a_tela_de_baralho(report):
    """Estado achado observando o jogo ao vivo (ADR-037), invisível nos frames
    salvos até então."""
    assert report.states["deck"] >= 1


def test_telas_de_escolha_pedem_modelo(report):
    """Diálogos são o que sobra pro VLM — se fosse zero, o roteamento estaria errado."""
    assert report.needed_model > 0
