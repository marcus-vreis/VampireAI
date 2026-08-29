# Estados do jogo

12 estados. O detector roda a cada captura e roteia para um handler. Input é
gamepad virtual (vgamepad → ViGEm → DualShock 4 emulado).

## Detecção: CV primeiro, VLM só no que sobra

Antes a detecção era 100% VLM. Era a chamada mais frequente do agente (95 de 319
no log) e a menos confiável: dos 39 frames que são o mapa, o modelo rotulou ao
menos 9 como outra coisa — a ponto de rodar um scan de cartas de 4 passos em cima
do mapa, apertando ← quatro vezes enquanto andava pela masmorra.

`src/vision/screen.py` mede três sinais e emite um veredito em ~19ms:

| Sinal | Mapa | Combate | Diálogo |
|---|---|---|---|
| pergaminho no minimapa | 0.62-0.76 | ≤ 0.32 | 0.05-0.07 |
| painel ardósia no centro | < 0.03 | 0.017-0.052 | 0.56-0.63 |
| círculos de custo na mão | 0 | ≥ 1 | 0 |

A tela **"Baralho"** fica no meio do painel ardósia, em 0.181 — entre o máximo do
combate (0.052) e o mínimo do diálogo (0.56). Ela precisou de estado próprio
porque as cartas do deck também têm círculo de custo, então passava por combate e
o agente tentaria jogar carta ali. Achada observando o jogo ao vivo; o limiar
ainda repousa sobre **uma** amostra e merece confirmação.

As margens são grandes, então a separação é confiável. O veredito é um de:

- `MAP`, `COMBAT` → estado final, sem chamar modelo
- `DIALOG` → VLM escolhe entre level_up / chest / boss_chest / chest_card_target / shop
- `UNKNOWN` → VLM escolhe entre title / menu / game_over / stage_complete / game_complete
- `NOT_GAME` → a captura não pegou o jogo; levanta `NotTheGameError` em vez de chutar

O último caso é real: há frames salvos onde a captura pegou outra janela. Antes o
VLM inventava um estado para eles.

## Lista

| Estado | Quando | Como é detectado |
|---|---|---|
| `combat` | inimigos no topo, cartas embaixo | CV: círculos de custo presentes |
| `map` | visão 1ª pessoa com minimapa | CV: pergaminho visível |
| `level_up` | "Subiu de nível!" | CV → VLM (subgrupo diálogo) |
| `chest` | baú comum aberto | CV → VLM (subgrupo diálogo) |
| `chest_card_target` | "aplicar bônus em qual carta?" | CV → VLM (subgrupo diálogo) |
| `boss_chest` | baú pós-chefe | CV → VLM (subgrupo diálogo) |
| `shop` | cartas com preço em ouro | CV → VLM (subgrupo diálogo) |
| `stage_complete` | pá no chão, ícone de próxima fase | CV → VLM (subgrupo outros) |
| `game_complete` | tela "PARABÉNS" | CV → VLM (subgrupo outros) |
| `title` | logo do jogo | CV → VLM (subgrupo outros) |
| `menu` | abas Unlocks / Settings / Crawlers | CV → VLM (subgrupo outros) |
| `game_over` | tela de derrota | CV → VLM (subgrupo outros) |
| `deck` | tela "Baralho", o deck inteiro | CV: painel ardósia intermediário |

## Tratamento por estado

### `combat`

1. `scan_combat_hand` percorre a mão com ← lendo uma carta por passo.
   **Não recebe o total** — para sozinho quando o cursor volta a uma posição já
   vista. Isso resolve a oclusão: a carta selecionada sobe e cobre o círculo de
   custo da vizinha à direita, então contar num frame só subestima em 1.
2. Cada carta é identificada pelo `CardDB` (hash perceptual). Só há chamada ao
   VLM na primeira aparição de cada carta.
3. O VLM decide UMA jogada. `src/combat.py` valida contra mana e índices; se for
   ilegal, repergunta com o motivo. No 3º erro, joga pela regra de `jogo.md`.
4. Executor navega `alvo − cursor` passos e aperta X.

**Estratégia central:** ordem CRESCENTE de custo de mana (combo buffado). Tomos
vermelhos (custo 0-1) primeiro para ganhar mana.

### `map`

Sem chamada de modelo nenhuma. Três passos:

1. `minimap.locate` acha o minimapa pelo bloco de pergaminho e **exige ≥50% de
   pergaminho na caixa** — senão, num frame de combate, a arte das cartas passava
   por mapa e um círculo de custo azul virava a seta do jogador.
2. `minimap.read_minimap` extrai posição e direção do jogador (a seta azul é o
   único azul saturado ali) e a máscara de piso. Ícones entram no piso: são cinza
   136, abaixo do limiar, e sem isso viravam buracos que impediam o BFS de chegar
   no inimigo.
3. `nav.plan` escolhe o alvo — inimigo alcançável mais próximo, depois chefe,
   depois fronteira inexplorada — e o BFS devolve a direção do primeiro trecho,
   que vira uma micro-ação: girar (L2/R2) ou andar (↑).

O minimapa muda de zoom entre fases, então nada assume tamanho de célula: a busca
é em espaço de pixels e só a direção é usada.

**Ícones** (`vision/icons.py`) saem de template matching contra sprites reais.
Prioridade segundo `jogo.md`: limpar os inimigos menores fortalece o personagem
pro chefe. Bônus não vale desvio — só se estiver no caminho.

Inspecionar com:

```bash
python -m src.vision.debug frames/algum.png
```

### `level_up`, `chest`, `boss_chest`, `chest_card_target`

VLM lê as opções, estrategista escolhe, o índice é preso ao intervalo válido e o
executor navega até ele. Baú vazio: □ para sacar dinheiro.

### `stage_complete` / `game_complete` / `title` / `menu` / `game_over`

Ações fixas (↑ / X / □ / abortar).

## Telas sem handler

O jogo tem telas que nenhum dos 12 estados cobre — as duas confirmações que a
evolução de carta abre são o caso conhecido. `src/stall.py` é a rede genérica:
se a tela não muda por 2 passos, escalona X → □ → andar pra frente; esgotado,
aborta a run em vez de girar em falso.

Isso não substitui handler correto. Quando houver frame real da tela, escreva o
handler — a rede existe pra que a ausência dele não termine a run.

## Verificação

O conjunto de regressão vem de `python -m src.label`: você joga marcando o estado
real de cada tela, e a ferramenta grava junto o que a CV respondeu. `--summary`
mostra a taxa de concordância por estado.
