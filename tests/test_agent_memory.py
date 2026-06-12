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
