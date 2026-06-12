"""Memória persistente em notes.md com sumarização accordion.

Formato do arquivo:

    # Notas do agente

    ## Resumo (sumarizado)
    <bloco de resumo, sobrescrito a cada accordion>

    ## Eventos recentes
    - [ISO_TS] state=combat | evento livre
    - [ISO_TS] ...

Quando os eventos passarem de `max_events`, o bloco mais antigo é colapsado em
"Resumo" via VLM (`summarize_fn`) e os eventos mais novos permanecem sob
"Eventos recentes". Isso é o "accordion": expandido durante o jogo, comprimido
periodicamente.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from loguru import logger

from src.config import PATHS

_HEADER = "# Notas do agente"
_SUMMARY_H = "## Resumo (sumarizado)"
_EVENTS_H = "## Eventos recentes"
_DEFAULT_MAX_EVENTS = 40
_DEFAULT_KEEP_RECENT = 10


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Memory:
    """Wrapper sobre notes.md.

    `summarize_fn` recebe uma string (eventos antigos colados) e devolve o
    resumo a ser persistido. Tipicamente um wrapper sobre `ask_vlm`.
    """

    path: Path
    max_events: int = _DEFAULT_MAX_EVENTS
    keep_recent: int = _DEFAULT_KEEP_RECENT
    summarize_fn: Callable[[str], str] | None = None

    def _read(self) -> tuple[str, list[str]]:
        if not self.path.is_file():
            return "", []
        text = self.path.read_text(encoding="utf-8")
        summary = ""
        events: list[str] = []
        section: str | None = None
        for raw in text.splitlines():
            line = raw.rstrip()
            if line == _SUMMARY_H:
                section = "summary"
                continue
            if line == _EVENTS_H:
                section = "events"
                continue
            if section == "summary" and line and not line.startswith("#"):
                summary += (line + "\n")
            elif section == "events" and line.startswith("- "):
                events.append(line[2:])
        return summary.strip(), events

    def _write(self, summary: str, events: list[str]) -> None:
        parts = [_HEADER, "", _SUMMARY_H, summary.strip() or "_(vazio)_", "", _EVENTS_H]
        parts.extend(f"- {e}" for e in events)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(parts) + "\n", encoding="utf-8")

    def append(self, event: str, *, state: str | None = None) -> None:
        """Adiciona um evento. Aciona accordion se passar de max_events."""
        summary, events = self._read()
        prefix = f"[{_now_iso()}]"
        if state:
            prefix += f" state={state}"
        events.append(f"{prefix} | {event}")
        if len(events) > self.max_events:
            summary = self._collapse(summary, events)
            events = events[-self.keep_recent:]
        self._write(summary, events)

    def _collapse(self, prior_summary: str, events: list[str]) -> str:
        old = events[: -self.keep_recent]
        if not old:
            return prior_summary
        if self.summarize_fn is None:
            logger.debug("Sem summarize_fn — concatenando resumo bruto")
            joined = "\n".join(f"- {e}" for e in old[-20:])
            return (prior_summary + "\n" + joined).strip()
        body = (
            (f"Resumo anterior:\n{prior_summary}\n\n" if prior_summary else "")
            + "Eventos a sumarizar (cronologicamente):\n"
            + "\n".join(f"- {e}" for e in old)
        )
        try:
            new_summary = self.summarize_fn(body).strip()
            logger.info("Memory: accordion compactou {} eventos", len(old))
            return new_summary
        except Exception as exc:  # noqa: BLE001
            logger.warning("summarize_fn falhou: {} — mantendo resumo anterior", exc)
            return prior_summary

    def recent(self, n: int = 10) -> list[str]:
        _, events = self._read()
        return events[-n:]

    def summary(self) -> str:
        s, _ = self._read()
        return s

    def reset(self) -> None:
        if self.path.is_file():
            self.path.unlink()


def default_memory(summarize_fn: Callable[[str], str] | None = None) -> Memory:
    return Memory(path=PATHS.notes / "notes.md", summarize_fn=summarize_fn)
