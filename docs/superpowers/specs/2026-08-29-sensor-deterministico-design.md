# Sensor determinístico — design

**Data:** 2026-08-29
**Status:** aprovado
**Substitui:** ADR-017 (mapa por pergunta de direção), ADR-019 (k-vote de contagem)

## Problema

O agente funciona ponta-a-ponta mas joga mal. Diagnóstico com evidência:

1. **`hand_area` recorta a mão errada.** `(380, 460, 480, 260)` cobre menos da metade
   do leque real (`~(250,380)-(1030,720)`). Num frame de 6 cartas, corta fora a
   carta mais à esquerda e fatia ao meio a carta selecionada — que é a posição
   inicial do cursor. `scan_combat_hand` usa o mesmo crop, então a leitura de
   carta acontece sobre um pedaço dela. Causa raiz de "contagem dá errado" **e**
   de "leitura piorou".
2. **`card_scan.txt` aponta o lugar errado.** Pede o custo no "canto superior
   direito"; nos frames o círculo de custo está sempre no canto superior
   **esquerdo**.
3. **Detecção de estado erra nos dois sentidos.** Frames salvos comprovam:
   `20260802T154240385_combat_initial.png` é o mapa;
   `20260802T154006402_map.png` é combate com 6 cartas.
4. **Erro acumula.** `agent.py` fixa `cursor_after_scan = 0`. Se `total` veio
   inflado, o cursor não está em 0 e todo índice seguinte sai deslocado.
5. **Não há como medir.** `llm.py` loga `raw_chars`, não a resposta.
6. **O alinhamento da captura escorrega entre sessões.** Regiões absolutas são
   medidas contra um referencial instável (barra de título presente ou não).

## Decisão

Mover percepção geométrica para CV determinística. O VLM fica só com semântica.

| Sinal | Antes | Depois |
|---|---|---|
| Estado da tela | VLM (95 chamadas, ~92%) | CV por assinatura |
| Contagem de cartas | VLM ×3 (k-vote) | CV: círculos de custo |
| Índice do cursor | VLM | CV: círculo maior |
| Mana / HP | VLM | CV + OCR |
| Direção no mapa | VLM ×4-8 por nó | BFS sobre grade do minimapa |
| Identidade da carta | VLM toda vez | VLM 1ª vez, cache por hash |
| Escolha da jogada | VLM | VLM **validado** por código |

### Trava de legalidade

O VLM continua decidindo a jogada. O código valida antes de executar: mana
suficiente, índice dentro da mão, carta existe. Ilegal → rejeita e repergunta;
no 3º erro cai na regra de custo crescente documentada em `jogo.md`.

Isso torna jogada ilegal impossível sem tirar a decisão do modelo, e a taxa de
rejeição vira métrica de qualidade para a fase de pesquisa.

## Arquitetura

```
src/window.py      localiza client area da janela via Win32 (ctypes)
src/capture.py     captura essa área  → frame estável 1280x720
src/vision/
  regions.py       regiões normalizadas ao client area
  cards.py         círculos de custo → total, cursor, bbox por carta
  hud.py           mana, HP
  minimap.py       minimapa → grade de células + features nas arestas
  screen.py        classificação de estado por assinatura CV
src/nav.py         BFS sobre a grade → sequência de micro-ações
src/combat.py      validador de jogada + fallback por regra
src/carddb.py      cache de identidade de carta por hash perceptual
src/label.py       ferramenta de captura rotulada (dataset de regressão)
```

`perceive()` mantém a assinatura. A troca é interna.

### Grade do minimapa

O minimapa é pixel art em grade fixa. Extraímos:
- células andáveis (bege claro) vs desconhecido (tan escuro) vs parede;
- posição **e direção** do jogador (seta azul — única cor saturada);
- features por ícone: caveira (inimigo), caveira com chifres (chefe), `?`,
  baú, pedra (obstáculo), ponto (bônus).

**Features de aresta:** pontos e baús encostados em parede são renderizados
deslocados para a borda da célula, não centralizados. A estrutura é grade de
células **+ conjunto de features presas a arestas** — não uma matriz 2D simples.

### Navegação

BFS da célula atual até o alvo → caminho de células → sequência de micro-ações
respeitando a direção atual (girar com L2/R2, andar com ↑). Sem chamada de VLM.

Alvo por prioridade: inimigos restantes → chefe. Bônus só se estiver no caminho.

## Verificação

Ferramenta de captura rotulada (`src/label.py`): o usuário joga apertando uma
tecla por frame para marcar o estado real. Saída: `dataset/` com frames +
`labels.jsonl`. Vira suíte de regressão em `tests/test_vision_regression.py`.

Métrica alvo: acurácia de estado ≥98%, contagem de cartas ≥95%, cursor ≥98%.

## Consequências

- Entram `numpy` + `opencv-python-headless`. Contra o "setup leve" do README,
  mas o protótipo em PIL puro custava ~200ms/frame; com numpy vai a ~5ms.
- ADR-017 e ADR-019 são aposentadas.
- Latência de turno de combate: de ~6+N chamadas VLM para ~1-2.

## Fora de escopo

Shop (raro), `?` do mapa (ignorado por ora), treino de modelo próprio.
