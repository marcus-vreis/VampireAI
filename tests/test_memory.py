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


def test_accordion_sem_llm_nao_cresce_sem_limite(tmp_path):
    """O fallback sem LLM ACRESCENTAVA 20 linhas a cada colapso e nunca encolhia.

    Um notes.md de sessão longa chegou a 254 mil caracteres, que `_memory_block`
    injetava inteiro em toda decisão de combate — ~63 mil tokens.
    """
    mem = Memory(path=tmp_path / "n.md", max_events=10, keep_recent=3)
    tamanhos = []
    for i in range(150):
        mem.append(f"evento {i} com texto de tamanho realista pra medir crescimento")
        if i % 30 == 29:
            tamanhos.append(len(mem.summary()))
    assert max(tamanhos) - min(tamanhos) < 500, f"resumo cresceu: {tamanhos}"


def test_resumo_gigante_herdado_e_cortado_na_leitura(tmp_path):
    """Arquivo de uma versão anterior não pode estourar o prompt de quem só lê."""
    path = tmp_path / "n.md"
    enorme = "\n".join(f"- linha antiga {i}" for i in range(5000))
    path.write_text(
        f"# Notas do agente\n\n## Resumo (sumarizado)\n{enorme}\n\n## Eventos recentes\n- x\n",
        encoding="utf-8",
    )
    resumo = Memory(path=path).summary()
    assert len(resumo) < 5000
    assert "descartado" in resumo


def test_corte_mantem_o_fim_do_resumo(tmp_path):
    """O mais recente é o que orienta a próxima decisão."""
    path = tmp_path / "n.md"
    linhas = "\n".join(f"- linha {i}" for i in range(200))
    path.write_text(
        f"# Notas do agente\n\n## Resumo (sumarizado)\n{linhas}\n\n## Eventos recentes\n- x\n",
        encoding="utf-8",
    )
    resumo = Memory(path=path).summary()
    assert "linha 199" in resumo
    assert "linha 0" not in resumo


def test_resumo_do_llm_tambem_e_cortado(tmp_path):
    """O modelo pode devolver um texto longo; o teto vale pros dois caminhos."""
    mem = Memory(
        path=tmp_path / "n.md",
        max_events=5,
        keep_recent=2,
        summarize_fn=lambda _: "\n".join(f"- bullet {i}" for i in range(400)),
    )
    for i in range(8):
        mem.append(f"evento {i}")
    assert len(mem.summary().splitlines()) <= 26
