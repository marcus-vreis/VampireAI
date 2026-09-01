"""Nenhum texto imprimivel pode sair do cp1252.

O console do Windows usa cp1252, e um caractere fora dele levanta
UnicodeEncodeError na hora de imprimir. Ja aconteceu QUATRO vezes neste projeto;
a quarta abortou o relatorio da primeira run de verdade do agente, depois da run
inteira ter acontecido -- o trabalho foi feito e o resultado se perdeu no print.

A regra estava so no CLAUDE.md, e regra sem teste e sugestao. Este teste faz a
distincao que importa: comentario e docstring nunca sao impressos e podem usar
seta e box-drawing a vontade; literal de string pode acabar num logger, num
print ou num help de argparse, e nao pode.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"


def _docstrings(arvore: ast.AST) -> set[int]:
    """id() dos nos que sao docstring de modulo, classe ou funcao."""
    marcados: set[int] = set()
    for no in ast.walk(arvore):
        corpo = getattr(no, "body", None)
        if not isinstance(no, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not corpo or not isinstance(corpo[0], ast.Expr):
            continue
        primeiro = corpo[0].value
        if isinstance(primeiro, ast.Constant) and isinstance(primeiro.value, str):
            marcados.add(id(primeiro))
    return marcados


def _fora_do_cp1252(texto: str) -> str:
    return "".join(sorted({c for c in texto if c.encode("cp1252", "ignore") == b""}))


@pytest.mark.parametrize("arquivo", sorted(_SRC.rglob("*.py")), ids=lambda p: p.name)
def test_literais_de_string_cabem_no_cp1252(arquivo: Path):
    arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
    docs = _docstrings(arvore)
    problemas = []
    for no in ast.walk(arvore):
        if not isinstance(no, ast.Constant) or not isinstance(no.value, str):
            continue
        if id(no) in docs:
            continue  # docstring nao e impressa
        ruins = _fora_do_cp1252(no.value)
        if ruins:
            problemas.append(f"linha {no.lineno}: {[hex(ord(c)) for c in ruins]}")
    assert not problemas, (
        f"{arquivo.name} tem literal com caractere fora do cp1252 "
        f"(use '->' no lugar da seta, '-' no lugar de box-drawing): {problemas}"
    )
