"""Modelos pydantic das saídas do VLM e das decisões do agente.

Convenção: cartas/opções são selecionadas por ÍNDICE na ordem da mão (0 = mais à
esquerda). Navegação concreta no gamepad é responsabilidade do executor.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ---------- Percepção: estados estruturados ----------


_MAX_MANA = 9


class CardScanFrame(BaseModel):
    """Saída do scan sequencial: 1 carta isolada (a 'em destaque') por frame."""
    nome: str
    mana: int | None = None
    descricao: str | None = None
    tipo: Literal["ataque", "tomo", "armadura", "utilitario", "bonus", "?"] = "?"

    @field_validator("mana")
    @classmethod
    def _mana_plausivel(cls, value: int | None) -> int | None:
        """Custo fora de 0..9 vira None em vez de propagar.

        O modelo já devolveu `mana=-1` lendo uma carta de bônus, cujo "-1" é o
        efeito ("custo reduzido em 1"), não o custo. Deixar passar faria
        `combat.validate` aprovar como se coubesse em qualquer mana.
        """
        if value is None or 0 <= value <= _MAX_MANA:
            return value
        return None


class StateDetection(BaseModel):
    estado: str


# ---------- Decisão: ações do agente ----------


class CombatAction(BaseModel):
    """Próxima ação em combate. Modelo decide UMA carta por vez ou encerra turno."""
    acao: Literal["jogar_carta", "finalizar_turno"]
    indice_alvo: int | None = Field(
        default=None,
        description="Índice (0-based) da carta a jogar. Obrigatório se acao=jogar_carta.",
    )
    motivo: str = Field(description="Justificativa curta (combo de mana, custo, etc.)")


class ChoiceAction(BaseModel):
    """Escolha em level_up / chest / bonus de carta. Navegação por índice."""
    indice_alvo: int = Field(description="Índice 0..N-1 da opção desejada")
    motivo: str


class LevelUpOption(BaseModel):
    posicao: int
    nome: str
    descricao: str | None = None
    mana: int | None = None
    e_bonus: bool = Field(
        default=False,
        description="True para carta de bônus (sem custo de mana, mais brilhante)",
    )


class LevelUpState(BaseModel):
    opcoes: list[LevelUpOption]
    indice_selecionada: int | None = None


class ShopItem(BaseModel):
    posicao: int
    nome: str
    preco: int | None = None
    descricao: str | None = None


class ShopState(BaseModel):
    itens: list[ShopItem]
    ouro: int | None = None


class ChestState(BaseModel):
    tipo: Literal["carta", "bonus", "evolucao", "vazio"] = "vazio"
    opcoes: list[LevelUpOption] = Field(default_factory=list)
    indice_selecionada: int | None = Field(
        default=None,
        description="Índice (0-based) da opção em destaque, se houver cursor visível",
    )
    recompensa_nome: str | None = None
    recompensa_descricao: str | None = None
