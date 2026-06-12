# Estados do jogo

Vampire Crawlers tem ~12 estados distinguíveis. Detector roda em cada captura e roteia para handler específico. Input é via **gamepad virtual** (vgamepad → ViGEm → DualShock 4 emulado), não mouse.

## Lista
| Estado | Quando | Identificadores visuais |
|---|---|---|
| `title` | Tela inicial | Logo "Vampire Crawlers", botão "Jogar" |
| `map` | Visão 1ª pessoa de masmorra | Mini-mapa de tabuleiro no canto, parede/corredor à frente |
| `combat` | Combate ativo | Inimigos no topo, cartas na barra inferior; uma carta em DESTAQUE |
| `level_up` | Após vitória | 3 cartas grandes, uma em destaque |
| `chest` | Baú aberto comum | Cartas / bônus / evolução / mensagem vazia |
| `chest_card_target` | Tela secundária do baú | "Aplicar bônus em qual carta do deck?" |
| `boss_chest` | Baú pós-chefe | UI igual a chest, contexto pós-combate de boss |
| `stage_complete` | Pá no chão à frente, ícone próxima fase no mapa | Fim de fase |
| `game_complete` | Tela "PARABÉNS" | Jogo zerado |
| `game_over` | Morte | Tela de derrota |
| `menu` | Pausa | Overlay |
| `shop` | Loja (raro) | Cartas com preço em ouro |

## Detecção
1ª versão: VLM com prompt de múltipla escolha (`src/prompts/detect_state.txt`). Sem template matching ainda. Latência aceita: ~3s por chamada.

## Tratamento por estado (gamepad)

### `combat`
1. Capture frame inicial → percepção (`CombatState`: total_cartas, cursor, mana, inimigos).
2. **Scan sequencial**: cursor inicia mais à direita. Para cada carta: pegue print, classifique (`CardScanFrame`), aperte ←. Repete `total_cartas` vezes.
3. Modelo decide UMA ação (`CombatAction`: jogar carta idx N, ou finalizar turno) com base no scan completo.
4. Executor calcula delta = `idx_alvo - cursor_atual`, navega N passos (← ou →), aperta X.
5. Retira essa carta da mão, recaptura, repete (mas mana atual mudou; cursor pode ter resetado).

**Estratégia central:** seguir ordem CRESCENTE de custo de mana (combo buffado). Tomos vermelhos (custo 0) primeiro pra ganhar mana.

### `map`
Pergunta visual mínima: "alvo está à frente, esquerda, direita ou atrás?" (`MapDirection`).
Micro-ações:
- frente → ↑ (D-pad)
- esquerda → L2 (girar)
- direita → R2 (girar)
- atrás → R2 + R2
- no_alvo → ↑ (entrar na sala)

Repete até `no_alvo`. Erros não acumulam (cada step recaptura).

Custo: ~4-8 chamadas VLM por nó × 3-5s ≈ 15-40s. Aceita porque mapa não é caminho crítico.

### `level_up`
Percepção das opções (até 3) → estrategista escolhe → navega → X.

### `chest` / `boss_chest`
Detecta tipo (carta / bônus / evolução / vazio). Vazio: aperta □ (cancel/sacar). Cartas: escolhe uma, X. Bônus abre `chest_card_target`. Evolução pede 2 cartas em sequência.

### `chest_card_target`
Prompt dedicado em `src/prompts/chest_card_target.txt` (separado do baú principal porque as duas telas têm estruturas diferentes — ver ADR-021). Percepção do deck mostrado + cursor visível → estrategista escolhe carta-alvo → navega delta passos → X. O delta usa `data["indice_selecionada"]` quando disponível.

### `stage_complete`
Anda pra frente (D-pad ↑) — passa pela pá, próxima fase.

### `game_complete`
Aperta X pra voltar ao menu. Encerra a run com vitória.

### `title` / `menu` / `game_over`
Ações fixas (X / □ / abortar).

## Convenção de prompts
Cada estado tem prompt em `src/prompts/{estado}.txt`. Schemas em `src/schemas.py`.
