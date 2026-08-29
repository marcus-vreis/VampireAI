"""Testes da injeção de memória nos prompts de decisão (`_memory_block`)."""

from __future__ import annotations

from pathlib import Path

from src.agent import _MEMORY_RECENT_EVENTS, _memory_block
from src.memory import Memory


def test_memory_block_none_returns_empty() -> None:
    assert _memory_block(None) == ""


def test_memory_block_empty_memory_returns_empty(tmp_path: Path) -> None:
    mem = Memory(path=tmp_path / "notes.md")
    assert _memory_block(mem) == ""


def test_memory_block_contains_recent_events(tmp_path: Path) -> None:
    mem = Memory(path=tmp_path / "notes.md", max_events=100)
    mem.append("entrou em combate", state="combat")
    mem.append("jogou tomo vazio", state="combat")
    block = _memory_block(mem)
    assert "MEMÓRIA DA RUN" in block
    assert "entrou em combate" in block
    assert "jogou tomo vazio" in block


def test_memory_block_caps_recent(tmp_path: Path) -> None:
    mem = Memory(path=tmp_path / "notes.md", max_events=100)
    for i in range(20):
        mem.append(f"evento {i}")
    block = _memory_block(mem)
    # cap = _MEMORY_RECENT_EVENTS, então só os 8 últimos aparecem
    for i in range(20 - _MEMORY_RECENT_EVENTS, 20):
        assert f"evento {i}" in block
    assert f"evento {20 - _MEMORY_RECENT_EVENTS - 1}" not in block


def test_memory_block_includes_summary(tmp_path: Path) -> None:
    mem = Memory(
        path=tmp_path / "notes.md",
        max_events=3,
        keep_recent=1,
        summarize_fn=lambda body: "resumo: jogador subiu 1 nível",
    )
    for i in range(4):
        mem.append(f"e{i}")
    block = _memory_block(mem)
    assert "Resumo:" in block
    assert "resumo: jogador subiu 1 nível" in block


def test_hp_alto_nao_gera_alerta():
    from src.agent import _hp_line
    from src.perception import HandScan

    assert _hp_line(HandScan(cards=[], cursor_idx=0, hp=(61, 61))) == "HP: 61/61"


def test_hp_baixo_avisa_o_modelo():
    """Sem o aviso, o modelo escolhe dano por padrão mesmo à beira da morte."""
    from src.agent import _hp_line
    from src.perception import HandScan

    linha = _hp_line(HandScan(cards=[], cursor_idx=0, hp=(18, 61)))
    assert "18/61" in linha
    assert "armadura" in linha


def test_hp_desconhecido_nao_polui_o_prompt():
    from src.agent import _hp_line
    from src.perception import HandScan

    assert _hp_line(HandScan(cards=[], cursor_idx=0)) == ""


def _scan(**kw):
    from src.perception import HandScan
    from src.schemas import CardScanFrame

    base = {
        "cards": [CardScanFrame(nome="Otto", mana=2, descricao="Cause 374.", tipo="ataque")],
        "cursor_idx": 0,
        "mana": 3,
    }
    return HandScan(**{**base, **kw})


def test_prompt_de_combate_carrega_mana_e_mao():
    from src.agent import _combat_prompt

    prompt = _combat_prompt(_scan(), None, None)
    assert "MANA DISPONÍVEL: 3" in prompt
    assert "Otto" in prompt
    assert '"indice": 0' in prompt


def test_prompt_de_combate_carrega_o_hp():
    """O HP era calculado mas não chegava ao prompt — testar a função isolada não pegava."""
    from src.agent import _combat_prompt

    assert "HP: 18/61" in _combat_prompt(_scan(hp=(18, 61)), None, None)


def test_prompt_de_combate_repassa_a_recusa():
    from src.agent import _combat_prompt

    prompt = _combat_prompt(_scan(), None, "a carta custa 2 e você tem 1")
    assert "rejeitada" in prompt
    assert "custa 2" in prompt


def _linhas(*eventos):
    return [f"[2026-01-01T00:00:00+00:00] state={s} | {t}" for s, t in eventos]


def test_bloco_ignora_ruido_de_mapa_e_destravamento():
    """Medido num prompt real: 15 eventos de mapa e 8 de destravamento contra
    ZERO de combate. Eram 578 tokens disputando espaço com a mão de cartas."""
    from src.agent import _relevant

    ruido = _linhas(
        ("map", "mapa: frente rumo a inimigo mais próximo"),
        ("stall", "destravando com confirm"),
        ("deck", "transição → deck"),
    )
    assert _relevant(ruido) == []


def test_bloco_mantem_jogadas_e_escolhas():
    from src.agent import _relevant

    util = _linhas(
        ("combat", "combate: jogou Otto — combo crescente"),
        ("level_up", "level up: idx=0 (sinergia com o deck)"),
        ("chest", "baú carta: idx=1"),
    )
    assert _relevant(util) == util


def test_transicao_nao_informa_decisao():
    """"transição → combat" não ajuda a escolher carta."""
    from src.agent import _relevant

    assert _relevant(_linhas(("combat", "transição → combat"))) == []


def test_prosa_do_sumarizador_passa():
    """Linha sem marca de estado é resumo do modelo, que já condensou tudo."""
    from src.agent import _relevant

    prosa = ["- Venceu dois combates e perdeu 12 de HP"]
    assert _relevant(prosa) == prosa
