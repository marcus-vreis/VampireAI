"""Ablação: quanto se perde ao devolver a percepção ao modelo.

A tese que motivou o pivô (ADR-022) é que CV determinística vence o VLM nas
perguntas de geometria. Isso estava sustentado por anedota — "de 39 frames que são
o mapa, ao menos 9 foram rotulados errado" — colhida de nomes de arquivo, não de
uma medição controlada.

Aqui os dois caminhos respondem à MESMA pergunta sobre os MESMOS frames, com
gabarito conferido olhando cada imagem:

- **cv**: `vision.screen.signature`, o caminho atual.
- **vlm**: o prompt de tela inteira que a ADR-022 aposentou, recuperado do
  histórico do git em `src/prompts/ablation/`.

O prompt aposentado fica sob `ablation/` de propósito: separado dos que o agente
usa, pra ninguém confundir código vivo com material de experimento.

    python -m src.ablation --repeticoes 3
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import time
from pathlib import Path

import cv2
from loguru import logger

from src.config import PATHS, PROJECT_ROOT
from src.llm import ask_vlm
from src.schemas import StateDetection
from src.vision.screen import signature

REFERENCIA = PROJECT_ROOT / "dataset" / "referencia"
LABELS = REFERENCIA / "labels.json"
_PROMPT_VLM = PATHS.prompts / "ablation" / "detect_state_vlm.txt"

# A CV agrupa as telas de escolha num veredito só; o VLM responde o estado exato.
# Comparar exige trazer os dois pro mesmo vocabulário: o grupo.
_GRUPO = {
    "level_up": "dialog",
    "chest": "dialog",
    "boss_chest": "dialog",
    "chest_card_target": "dialog",
    "shop": "dialog",
    "title": "outro",
    "menu": "outro",
    "game_over": "outro",
    "stage_complete": "outro",
    "game_complete": "outro",
}


def _grupo(estado: str) -> str:
    return _GRUPO.get(estado, estado)


def opcoes_do_prompt() -> set[str]:
    """Estados que o prompt aposentado oferecia como resposta.

    Importa pra ser justo: `deck` e `not_game` foram descobertos DEPOIS que ele
    saiu de uso, então ele não tinha como acertá-los. Contar isso como erro do
    modelo seria trapaça na comparação.
    """
    prompt = _PROMPT_VLM.read_text(encoding="utf-8")
    return {_grupo(m) for m in re.findall(r'"([a-z_]+)"', prompt)} - {"estado"}


def carregar_gabarito() -> dict[str, str]:
    dados = json.loads(LABELS.read_text(encoding="utf-8"))
    return dados["frames"]


def responder_cv(frame) -> tuple[str, float]:
    inicio = time.perf_counter()
    veredito = signature(frame).verdict
    return veredito.value, time.perf_counter() - inicio


def responder_vlm(caminho: Path) -> tuple[str, float]:
    prompt = _PROMPT_VLM.read_text(encoding="utf-8")
    inicio = time.perf_counter()
    try:
        r = ask_vlm(str(caminho), prompt, schema=StateDetection)
        estado = str(r.get("estado", "?")).strip().lower()
    except Exception as e:  # noqa: BLE001 - falhar é um resultado da ablação
        logger.debug("vlm falhou em {}: {}", caminho.name, e)
        estado = "(erro)"
    return estado, time.perf_counter() - inicio


def rodar(repeticoes: int) -> dict:
    gabarito = carregar_gabarito()
    expressaveis = opcoes_do_prompt()
    vazio = {"acertos": 0, "total": 0, "tempos": [], "erros": [], "just_ok": 0, "just_n": 0}
    resultado: dict = {"cv": dict(vazio, erros=[], tempos=[]), "vlm": dict(vazio, erros=[], tempos=[])}
    for nome, esperado in gabarito.items():
        caminho = REFERENCIA / nome
        if not caminho.is_file():
            continue
        frame = cv2.imread(str(caminho))
        alvo = _grupo(esperado)

        exprimivel = alvo in expressaveis

        estado, dt = responder_cv(frame)
        _pontuar(resultado["cv"], nome, _grupo(estado), alvo, dt, exprimivel)

        for _ in range(repeticoes):
            estado, dt = responder_vlm(caminho)
            _pontuar(resultado["vlm"], nome, _grupo(estado), alvo, dt, exprimivel)
    resultado["_fora_do_vocabulario"] = sorted(
        {e for e in gabarito.values() if _grupo(e) not in expressaveis}
    )
    return resultado


def _pontuar(
    acc: dict, nome: str, obtido: str, esperado: str, dt: float, justo: bool
) -> None:
    acc["total"] += 1
    acc["tempos"].append(dt)
    acertou = obtido == esperado
    acc["acertos"] += acertou
    if justo:
        acc["just_n"] += 1
        acc["just_ok"] += acertou
    if not acertou:
        marca = "" if justo else "  (estado fora do vocabulário do prompt)"
        acc["erros"].append(f"{nome[:34]}: disse {obtido}, era {esperado}{marca}")


def relatorio(resultado: dict) -> str:
    linhas = [
        "",
        f"{'caminho':8} {'geral':>16} {'só o expressável':>18} {'mediana':>11}",
        "-" * 58,
    ]
    for nome in ("cv", "vlm"):
        acc = resultado[nome]
        if not acc["total"]:
            continue
        taxa = 100 * acc["acertos"] / acc["total"]
        justa = 100 * acc["just_ok"] / acc["just_n"] if acc["just_n"] else 0.0
        linhas.append(
            f"{nome:8} {acc['acertos']:>3}/{acc['total']:<3} ({taxa:>3.0f}%) "
            f"{acc['just_ok']:>3}/{acc['just_n']:<3} ({justa:>3.0f}%) "
            f"{statistics.median(acc['tempos']) * 1000:>8.0f} ms"
        )
    for nome in ("cv", "vlm"):
        erros = collections.Counter(resultado[nome]["erros"])
        if not erros:
            continue
        linhas.append(f"\nerros de {nome}:")
        linhas.extend(f"  {n}x {e}" if n > 1 else f"  {e}" for e, n in erros.most_common(8))
    fora = resultado.get("_fora_do_vocabulario")
    if fora:
        linhas.append(
            f"\nfora do vocabulário do prompt aposentado: {', '.join(fora)}"
            "\ndescobertos DEPOIS que ele saiu de uso. A coluna 'só o expressável'"
            "\nexclui esses frames, que ele não tinha como acertar."
        )
    return "\n".join(linhas)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compara CV e VLM na mesma pergunta.")
    parser.add_argument(
        "--repeticoes", type=int, default=3,
        help="Quantas vezes perguntar ao modelo por frame. A CV é determinística.",
    )
    args = parser.parse_args()

    if not LABELS.is_file():
        logger.error("Gabarito não encontrado em {}", LABELS)
        return 2
    resultado = rodar(args.repeticoes)
    print(relatorio(resultado))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
