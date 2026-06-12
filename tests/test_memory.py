"""Testes da memória persistente (accordion)."""

from __future__ import annotations

from pathlib import Path

from src.memory import Memory


def test_append_and_recent(tmp_path: Path) -> None:
    mem = Memory(path=tmp_path / "notes.md", max_events=100)
    mem.append("entrou em combate", state="combat")
    mem.append("jogou carta 0", state="combat")
    recent = mem.recent(5)
    assert len(recent) == 2
    assert "entrou em combate" in recent[0]
    assert "state=combat" in recent[0]


def test_accordion_collapse_without_llm(tmp_path: Path) -> None:
    mem = Memory(path=tmp_path / "notes.md", max_events=5, keep_recent=2)
    for i in range(6):
        mem.append(f"evento {i}")
    summary = mem.summary()
    assert summary
    assert "evento" in summary
    recent = mem.recent(10)
    assert len(recent) == 2
    assert "evento 5" in recent[-1]


def test_accordion_uses_summarize_fn(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_summarize(body: str) -> str:
        calls.append(body)
        return "RESUMO_FAKE"

    mem = Memory(
        path=tmp_path / "notes.md",
        max_events=3,
        keep_recent=1,
        summarize_fn=fake_summarize,
    )
    for i in range(4):
        mem.append(f"e{i}")
    assert calls, "summarize_fn deveria ter sido chamada"
    assert mem.summary() == "RESUMO_FAKE"
    assert len(mem.recent(10)) == 1
    assert "e3" in mem.recent(10)[0]


def test_persistence_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    Memory(path=path).append("a", state="map")
    Memory(path=path).append("b", state="combat")
    events = Memory(path=path).recent(10)
    assert len(events) == 2
    assert "a" in events[0] and "b" in events[1]


def test_reset(tmp_path: Path) -> None:
    mem = Memory(path=tmp_path / "notes.md")
    mem.append("x")
    mem.reset()
    assert mem.recent(5) == []
