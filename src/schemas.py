"""Modelos pydantic das saídas do VLM e das decisões do agente.

Convenção: cartas/opções são selecionadas por ÍNDICE na ordem da mão (0 = mais à
esquerda). Navegação concreta no gamepad é responsabilidade do executor.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------- Percepção: estados estruturados ----------


class Card(BaseModel):
    posicao: int = Field(description="Índice na mão, 0 = mais à esquerda")
    nome: str
    mana: int | None = Field(default=None, description="Custo de mana, ? → null")
    descricao: str | None = None
    obstruida: bool = False
    selecionada: bool = Field(
        default=False,
        description="True se esta carta está em destaque/maior (cursor sobre ela)",
    )


class Enemy(BaseModel):
    posicao: int
    hp: int | None = None
    nome: str | None = None


class CombatState(BaseModel):
    """Estado mínimo do combate. Detalhes de cartas vêm via card_scan; HP/inimigos
    são opcionais e ficam pra OCR/visão posterior se necessário."""
    total_cartas_visiveis: int
    indice_selecionada: int | None = Field(
        default=None,
        description="Índice 0..N-1 da carta atualmente em destaque",
    )
    mana: int | None = None
    cartas: list[Card] = Field(default_factory=list)
    mana_max: int | None = None
    hp_player: int | None = None
    hp_max: int | None = None
    inimigos: list[Enemy] = Field(default_factory=list)


class CardScanFrame(BaseModel):
    """Saída do scan sequencial: 1 carta isolada (a 'em destaque') por frame."""
    nome: str
    mana: int | None = None
    descricao: str | None = None
    tipo: Literal["ataque", "tomo", "armadura", "utilitario", "bonus", "?"] = "?"


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


class MapDirection(BaseModel):
    """Resposta visual simples sobre o alvo no mapa 3D."""
    direcao_alvo: Literal["frente", "esquerda", "direita", "atras", "no_alvo"]
    motivo: str | None = None


class MapAction(BaseModel):
    """Micro-ação derivada da direção. O agente repete até chegar."""
    acao: Literal["andar_frente", "girar_esquerda", "girar_direita", "no_alvo"]


class ChoiceAction(BaseModel):
    """Escolha em level_up / chest / bonus de carta. Navegação por índice."""
    indice_alvo: int = Field(description="Índice 0..N-1 da opção desejada")
    motivo: str


class ChestAction(BaseModel):
    """Ação dentro de baú: escolher carta, sacar dinheiro, etc."""
    acao: Literal["escolher_carta", "sacar_dinheiro", "evoluir"]
    indice_alvo: int | None = None
    motivo: str | None = None


# ---------- Estados antigos mantidos por compat (level_up, map, shop, chest) ----------


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


class MapNode(BaseModel):
    id: str
    tipo: str
    disponivel: bool


class MapState(BaseModel):
    nos: list[MapNode]
    no_atual: str | None = None


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
