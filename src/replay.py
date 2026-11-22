"""Roda a máquina de estados contra frames salvos, sem jogo e sem gamepad.

O maior risco do projeto é que cada peça funcione isolada e o conjunto não. Todo
teste até aqui é unitário; o agente inteiro nunca rodou. Um bug real já escapou
por isso: o HP era lido corretamente, `_hp_line` tinha teste passando, e mesmo
assim o dado não chegava ao prompt — a linha nunca foi inserida na lista.

O replay fecha essa lacuna sem depender do jogo aberto: captura vira leitura de
arquivo, gamepad entra em dry-run e (com `--offline`) o modelo é bloqueado. O que
sobra exercitado é o que importa — percepção, roteamento de estado, handler e a
ação pretendida.

    python -m src.replay --frames frames --offline

Sem `--offline` as chamadas ao modelo acontecem de verdade e consomem GPU.
"""

from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

import cv2
from loguru import logger

from src import gamepad
from src.stall import StallDetector
from src.states import GameState, NotTheGameError


class OfflineError(RuntimeError):
    """Uma chamada de modelo foi tentada com `--offline` ligado."""


_STALL_MARKER = "tela travada"


def _safe(text: str) -> str:
    """Texto imprimível no console. cp1252 do Windows não aceita seta nem emoji."""
    return text.encode("ascii", "replace").decode("ascii")


@dataclass
class ReplayReport:
    states: collections.Counter = field(default_factory=collections.Counter)
    actions: list[str] = field(default_factory=list)
    needed_model: int = 0
    stalls: int = 0
    errors: list[str] = field(default_factory=list)
    frames: int = 0

    def summary(self) -> str:
        linhas = [
            f"{self.frames} frames processados",
            f"estados: {dict(self.states)}",
            f"ações pretendidas: {len(self.actions)}",
            f"telas que exigiram modelo: {self.needed_model}",
            f"travamentos detectados: {self.stalls}",
            f"erros: {len(self.errors)}",
        ]
        linhas.extend(f"  ERRO {_safe(e)}" for e in self.errors)
        return "\n".join(linhas)


def _frames_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.png") if not p.name.endswith("_debug.png"))


def _refuse(*_args, **_kwargs):
    raise OfflineError("chamada de modelo bloqueada por --offline")


def replay(folder: Path, offline: bool, limit: int | None = None) -> ReplayReport:
    """Percorre os frames alimentando o loop do agente.

    Liga o dry-run do gamepad antes de qualquer coisa: garantir isso no chamador
    não funcionou (ver comentário em `gamepad.set_dry_run`).
    """
    from src import agent

    gamepad.set_dry_run(True)
    report = ReplayReport()
    detector = StallDetector()
    memory = agent.default_memory()

    for path in _frames_in(folder)[:limit]:
        frame = cv2.imread(str(path))
        if frame is None or frame.shape[:2] != (720, 1280):
            continue
        report.frames += 1
        _run_one(agent, path, frame, detector, memory, report, offline)
    return report


def _patches(agent, path: Path, offline: bool) -> list:
    """Substitui captura e, em modo offline, toda chamada de modelo."""
    active = [
        mock.patch.object(agent, "grab", return_value=path),
        mock.patch("src.perception.grab", return_value=path),
    ]
    if offline:
        active += [
            mock.patch("src.states.ask_vlm", _refuse),
            mock.patch("src.perception.ask_vlm", _refuse),
            mock.patch("src.agent.ask_vlm", _refuse),
        ]
    return active


def _run_one(agent, path, frame, detector, memory, report, offline) -> None:
    before = len(memory.recent(200))
    stack = _patches(agent, path, offline)
    for patch in stack:
        patch.start()
    try:
        state = agent._step(memory, None, detector)
    except OfflineError:
        report.needed_model += 1
        return
    except NotTheGameError:
        report.states["not_game"] += 1
        return
    except RuntimeError as e:
        # Travamento não é falha do replay: é o detector fazendo o trabalho dele
        # sobre uma sequência em que o agente REALMENTE estava preso.
        bucket = report.errors if _STALL_MARKER not in str(e) else None
        if bucket is None:
            report.stalls += 1
        else:
            bucket.append(f"{path.name}: {type(e).__name__}: {e}")
        return
    except Exception as e:  # noqa: BLE001 - o replay existe pra catalogar falhas
        report.errors.append(f"{path.name}: {type(e).__name__}: {e}")
        return
    finally:
        for patch in reversed(stack):
            patch.stop()
        report.actions.extend(memory.recent(200)[before:])
    if isinstance(state, GameState):
        report.states[state.value] += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Roda o agente contra frames salvos.")
    parser.add_argument("--frames", default="frames", help="Pasta com os PNGs.")
    parser.add_argument(
        "--offline", action="store_true",
        help="Bloqueia toda chamada de modelo. Não consome GPU.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    folder = Path(args.frames)
    if not folder.is_dir():
        logger.error("Pasta não encontrada: {}", folder)
        return 2

    report = replay(folder, offline=args.offline, limit=args.limit)
    print(report.summary())
    if report.actions:
        print("\nprimeiras decisões registradas:")
        for action in report.actions[:12]:
            print(f"  {_safe(action)}")
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
