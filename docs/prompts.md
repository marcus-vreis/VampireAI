# Convenções de prompt

VLM local (Qwen2.5-VL 7B) requer prompts mais rígidos que API top-tier. Aprendizados consolidados aqui.

## Idioma
Sempre PT-BR. UI do jogo é em português; usar termos exatos.

## Regras universais
1. **Transcrição literal.** "transcreva exatamente como aparece, sem corrigir ortografia, traduzir ou substituir sinônimos". Sem isso, o modelo "corrige" Espinafre → Espinácea.
2. **Contagem rigorosa.** "conte cada carta visível, mesmo as parcialmente cobertas atrás de outras".
3. **Marcação de incerteza.** "use null para número ilegível, ? para texto ilegível". Melhor incerteza que alucinação.
4. **Schema explícito no prompt.** Sempre mostrar o JSON esperado no prompt.
5. **Sem prosa antes do JSON.** "Responda APENAS com o JSON".

## Padrões específicos do gamepad

### Combat — "carta selecionada"
A carta selecionada é a **maior / mais à frente / em destaque**. Cada prompt de combate exige `selecionada: true` para exatamente UMA carta, e `indice_selecionada` igual à posição dela.

### Card scan — uma carta isolada
`card_scan.txt` recebe um print onde a carta destacada ocupa quase a tela inteira. Prompt curtíssimo: nome, mana, descrição, tipo. Sem contexto da mão completa.

### Map — pergunta mínima
`map.txt` reduz o problema à pergunta mais fácil possível: "o alvo está à frente, esquerda, direita ou atrás?" (5 opções, incluindo `no_alvo`). Nada de contar paredes ou ler mini-mapa em detalhe.

### Decisão (combat_decide)
Prompt do estrategista é texto puro (sem imagem) que recebe JSON do estado + JSON do scan. Modelo devolve UMA ação. Princípio: erros não acumulam — próxima captura corrige.

## Estrutura padrão
```
[contexto + mecânica do gamepad]
[regras de transcrição literal]
[schema JSON inline]
[instrução "responda APENAS JSON"]
```

## Validação
Toda saída de VLM passa por pydantic (`src/schemas.py`). 3 falhas seguidas → abortar turno (`_MAX_PARSE_FAILS` em `agent.py`).

## Multi-amostra para contagem
`count_cards.txt` é chamado K vezes (`PERCEPTION_COUNT_SAMPLES`, default 3) sobre o mesmo crop pré-processado (contraste/saturação/nitidez via `ImageEnhance`). O total final é o **mode** das K amostras; o índice idem dentro do total vencedor. Com isso o erro ±1 do VLM cai pra <1% em cartas de 3-7. Ver `_vote_count` em `src/perception.py`.

Tunables (`PerceptionConfig`):
- `PERCEPTION_ENHANCE_CONTRAST` (1.25) — separa fundo do leque das cartas.
- `PERCEPTION_ENHANCE_SATURATION` (1.4) — destaca a bolinha cyan da mana.
- `PERCEPTION_ENHANCE_SHARPNESS` (1.2) — bordas dos números brancos.

Para inspecionar visualmente o realce: `python -m src.perception --frame F.png --crop hand_area --enhance` salva `frames/crop_hand_area_enhanced.png`.

## OCR auxiliar (status: pendente)
Após o pivô gamepad, OCR de HP/mana globais ficou no roadmap mas ainda não está
implementado em `perception.py`. As regiões em `OCR_REGIONS` permanecem como
placeholders (`(-1, -1, -1, -1)`). Hoje a percepção de combate usa apenas crops
para VLM (`hand_area`, `mana_orb`). HP do jogador não é lido — fica como TODO.

Cartas isoladas: VLM (porque o custo de mana fica grande no print da carta destacada).

## Memória injetada nas decisões
A partir de 2026-05-03, prompts de decisão (`combat_decide`, `level_up`, `chest`)
recebem o resumo accordion + últimos 8 eventos de `notes/notes.md`. Implementado
em `agent._memory_block`. Permite que a estratégia "preserve HP, evite cartas
redundantes" se acumule entre turnos.
