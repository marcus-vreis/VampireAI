# Prompts

Convenções e inventário. Ler antes de mexer em qualquer arquivo de
`src/prompts/`.

## Princípios

- **PT-BR.** A UI do jogo é em português; pedir tradução acrescenta um passo onde
  o modelo erra.
- **Transcrição literal.** Qwen2.5-VL "corrige" nomes que não conhece. Todo
  prompt de leitura diz explicitamente para não corrigir ortografia.
- **Schema no final, JSON puro.** Toda saída é validada por pydantic
  (`src/schemas.py`). Prompt sem schema explícito produz markdown em volta do
  JSON.
- **Uma pergunta por prompt.** Prompts que pedem várias coisas degradam em todas.

## O que o VLM ainda faz

Depois da ADR-022 a percepção geométrica saiu do modelo. Restaram:

| Prompt | Quando | Custo típico |
|---|---|---|
| `card_scan.txt` | carta que ainda não está no `CardDB` | 1 chamada por carta nova |
| `detect_dialog.txt` | CV disse "é tela de escolha", falta saber qual | raro |
| `detect_other.txt` | CV não reconheceu a tela | raro |
| `level_up.txt` | ler as opções de recompensa | 1 por level up |
| `chest.txt`, `chest_card_target.txt` | ler opções do baú | 1 por baú |
| `combat_decide.txt` | escolher a jogada | 1-3 por jogada (ver validação) |
| `read_mana_orb.txt` | só se o Tesseract não estiver instalado | 0 ou 1 por turno |

### Aposentados

- `count_cards.txt` — a contagem virou CV + travessia (ADR-019 e ADR-024
  aposentadas). O arquivo continua no repo para referência histórica.
- `map.txt` — a direção sai do BFS sobre o minimapa (ADR-017 aposentada).
- `detect_state.txt` — substituído pelo par `detect_dialog` / `detect_other`,
  que perguntam dentro de um subgrupo já restrito pela CV.

## Perguntas restritas

`detect_dialog.txt` e `detect_other.txt` existem porque a CV já eliminou a maior
parte do espaço de resposta antes de o modelo ser chamado. Perguntar "qual destas
5 telas de escolha é" acerta muito mais que "qual destes 12 estados é" — e o
prompt pode gastar seu espaço em dicas discriminativas em vez de listar opções.

## Correção importante

`card_scan.txt` pedia o custo no *"canto superior direito"*. Nos frames o círculo
de custo está sempre no canto superior **esquerdo**. O modelo procurava num lugar
vazio e inventava número. Corrigido em 2026-08-29.

## Memória injetada nas decisões

`combat_decide` e as escolhas de level up / baú recebem o resumo accordion +
últimos 8 eventos da `Memory`, formatados por `agent._memory_block`. Ver ADR-020.

## Validação da jogada

`combat_decide.txt` roda dentro de um laço (`agent._decide_combat`): se a carta
escolhida não cabe na mana ou o índice não existe, o prompt é reenviado com o
motivo da recusa anexado. Após 2 tentativas, o código joga pela regra de
`jogo.md`. Ver ADR-026.

Isso significa que **mexer neste prompt afeta a taxa de rejeição**, que é a
métrica de qualidade do modelo. Registrar o antes e depois ao alterar.
