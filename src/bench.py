"""Comparação de modelos na tarefa de decidir a jogada.

A pergunta "qual modelo usar?" não se responde por lista de blog. Aqui ela vira
número, e sem precisar de rotulagem humana: a legalidade de uma jogada é uma
regra que o código já conhece (mana suficiente, índice existente), então dá pra
gerar cenários e corrigir a prova sozinho.

Três medidas, da mais dura pra mais macia:

- **parse**: a resposta é JSON válido no schema? Falhar aqui trava o agente.
- **legal**: a carta escolhida é jogável? Ilegal força repergunta e queima tempo.
- **regra**: bate com a heurística de `jogo.md` (tomo mais barato primeiro, senão
  a carta jogável mais barata)? Não é gabarito absoluto — uma jogada fora da regra
  pode ser melhor — mas divergência sistemática indica que o modelo não entendeu
  a mecânica.

Uso:

    python -m src.bench --models qwen2.5vl:7b,outro-modelo --scenarios 20
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import dataclass

from loguru import logger

from src.combat import affordable, fallback_index, validate
from src.config import LLM, PATHS
from src.llm import ask_vlm
from src.schemas import CardScanFrame, CombatAction

# Cartas reais do jogo, transcritas de frames. Manter fiel: nome e efeito
# influenciam a decisão do modelo tanto quanto o custo.
_POOL = [
    ("Tomo Vazio", 0, "tomo", "Adicione 1 Mana."),
    ("Tomo Leve", 1, "tomo", "Adicione 2 Mana."),
    ("Armadura", 0, "armadura", "Adicione 2 de armadura."),
    ("Osso", 0, "ataque", "Cause 36 de dano a vários inimigos."),
    ("Espinafre", 1, "utilitario", "Cause 10% a mais de dano até o fim do combate."),
    ("Giovanna", 1, "utilitario", "Adicione 20 Sorte. Rastejante."),
    ("Gatti Amari", 1, "ataque", "Cause 102 de dano. Chance de Briga."),
    ("Otto, o Pardal", 2, "ataque", "Cause 374 de dano."),
    ("Phiera Der Tuphello", 3, "ataque", "Cause 258 de dano."),
    ("Pugnala", 1, "ataque", "Poder: Cause 20% a mais de dano. Rastejante."),
]


@dataclass
class Scenario:
    hand: list[CardScanFrame]
    mana: int

    @property
    def playable(self) -> list[int]:
        return affordable(self.hand, self.mana)

    @property
    def by_rule(self) -> int | None:
        return fallback_index(self.hand, self.mana)


@dataclass
class Tally:
    parsed: int = 0
    legal: int = 0
    by_rule: int = 0
    total: int = 0
    latencies: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.latencies = []

    def rate(self, hits: int) -> float:
        return 100.0 * hits / self.total if self.total else 0.0


def build_scenarios(n: int, seed: int) -> list[Scenario]:
    """Cenários com ao menos uma jogada legal — senão não há o que medir."""
    rng = random.Random(seed)
    out: list[Scenario] = []
    while len(out) < n:
        size = rng.randint(3, 6)
        hand = [
            CardScanFrame(nome=nome, mana=custo, descricao=desc, tipo=tipo)
            for nome, custo, tipo, desc in rng.sample(_POOL, size)
        ]
        scenario = Scenario(hand=hand, mana=rng.randint(0, 4))
        if scenario.playable:
            out.append(scenario)
    return out


def _prompt(scenario: Scenario) -> str:
    hand = [{"indice": i, **c.model_dump()} for i, c in enumerate(scenario.hand)]
    return "\n".join(
        [
            (PATHS.prompts / "combat_decide.txt").read_text(encoding="utf-8"),
            f"\nMANA DISPONÍVEL: {scenario.mana}",
            "\nMÃO (índices 0 = mais à esquerda):",
            json.dumps(hand, ensure_ascii=False, indent=2),
        ]
    )


def _score_one(model: str, scenario: Scenario, tally: Tally) -> None:
    started = time.monotonic()
    try:
        raw = ask_vlm(None, _prompt(scenario), schema=CombatAction, model=model)
    except Exception as e:  # noqa: BLE001 - falha de parse conta como erro do modelo
        logger.debug("{}: {}", model, e)
        tally.latencies.append(time.monotonic() - started)
        return
    tally.latencies.append(time.monotonic() - started)
    tally.parsed += 1

    action = CombatAction(**raw)
    if action.acao == "finalizar_turno":
        return  # havia jogada possível; encerrar não é ilegal, só não conta acerto
    if validate(action.indice_alvo, scenario.hand, scenario.mana) is None:
        tally.legal += 1
        if action.indice_alvo == scenario.by_rule:
            tally.by_rule += 1


def run(models: list[str], scenarios: list[Scenario]) -> dict[str, Tally]:
    results: dict[str, Tally] = {}
    for model in models:
        tally = Tally(total=len(scenarios))
        logger.info("Avaliando {} em {} cenários...", model, len(scenarios))
        for scenario in scenarios:
            _score_one(model, scenario, tally)
        results[model] = tally
    return results


def report(results: dict[str, Tally]) -> None:
    print(f"\n{'modelo':32} {'parse':>7} {'legal':>7} {'regra':>7} {'mediana':>9}")
    print("-" * 68)
    for model, t in results.items():
        median = statistics.median(t.latencies) if t.latencies else 0.0
        print(
            f"{model:32} {t.rate(t.parsed):>6.0f}% {t.rate(t.legal):>6.0f}% "
            f"{t.rate(t.by_rule):>6.0f}% {median:>8.2f}s"
        )
    print(
        "\nparse = JSON válido no schema | legal = cabe na mana e no índice"
        "\nregra = bate com a heurística de jogo.md (tomo barato primeiro)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compara modelos na decisão de combate.")
    parser.add_argument(
        "--models",
        default=f"{LLM.text_model}",
        help="Lista separada por vírgula. Default: o TEXT_MODEL configurado.",
    )
    parser.add_argument("--scenarios", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7, help="Mesma seed = mesmos cenários.")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    scenarios = build_scenarios(args.scenarios, args.seed)
    report(run(models, scenarios))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
