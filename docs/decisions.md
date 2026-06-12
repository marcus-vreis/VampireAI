# Decisões arquiteturais

Registro curto de escolhas feitas e o porquê. Estilo ADR simplificado.

## ADR-001: VLM local em vez de API
**Data:** 2026-05
**Decisão:** Qwen2.5-VL 7B via Ollama.
**Motivo:** custo zero (24/7 sem fatura), ângulo de pesquisa "open-weight pequeno", cabe em 16GB VRAM.
**Trade-off:** raciocínio menor — precisa prompts rígidos e OCR auxiliar.

## ADR-002: SDK openai apontando Ollama
**Decisão:** `openai` SDK com `base_url=http://localhost:11434/v1`.
**Motivo:** portabilidade. Trocar pra Claude/GPT é só mudar `base_url`.

## ADR-003: OCR híbrido para números pequenos
**Decisão:** pytesseract antes de VLM para HP/mana globais.
**Motivo:** VLM falha em números pequenos sobre sprites; OCR upscale 4× + threshold é mais previsível.

## ADR-004: Prompts em arquivos separados
**Decisão:** prompts em `src/prompts/{estado}.txt`.
**Motivo:** iteração sem mexer em código, ablation futura.

## ADR-005: Loguru
**Decisão:** loguru no lugar de logging stdlib.
**Motivo:** menos boilerplate.

## ADR-006: Pydantic para validação de saída
**Decisão:** todo retorno de VLM passa por pydantic.
**Motivo:** falha rápida em vez de KeyError difuso.

## ADR-008: Config via `.env` com fallback em `.env.example`
**Decisão:** `load_dotenv(.env)` + `load_dotenv(.env.example, override=False)`.
**Motivo:** clone novo roda sem copiar `.env` na mão.

## ADR-009: Detecção de estado via VLM
**Decisão:** prompt curto de múltipla escolha, sem heurística de pixel ainda.
**Motivo:** validar caminho ponta-a-ponta antes de otimizar.

## ADR-010: OCR vence VLM em divergência
**Decisão:** `_reconcile_combat` sobrescreve VLM quando OCR diverge.
**Motivo:** dígitos pequenos confundem VLM (8↔3, 5↔6).

## ADR-014: Pivot para gamepad virtual (2026-05-02) ⭐
**Data:** 2026-05-02
**Decisão:** abandonar pyautogui + coords de UI; usar `vgamepad` (Python) + driver ViGEm Bus, simulando um DualShock 4 no nível de driver.
**Motivo:**
- O jogo aceita controle nativamente (jogo.md).
- Elimina TODA calibração de UI: BUTTONS, HAND_AREA, card_slot_center, OCR_REGIONS dos slots de carta.
- Carta selecionada = "maior/em destaque". Navegação por travessia (← →) — robusto a redistribuição de mão.
- Sem ambiguidade quanto à correspondência ação ↔ botão.
**Trade-off aceito:**
- Dependência do driver ViGEm Bus (Windows-only, instalação manual uma vez).
- Sem `pyautogui.FAILSAFE` global; mitigação: gamepad só afeta janela focada do jogo, e `gamepad.reset()` no `finally`.
- Mapeamento de "finalizar turno" precisa observação manual (ver TODO em `input_exec.py`).
**Substitui:** ADR-011 (coords TBD), ADR-013 (slots calculados em runtime). Ambos obsoletos.

## ADR-015: Scan sequencial de cartas em combate
**Data:** 2026-05-02
**Decisão:** ao entrar em combate, perceber `total_cartas` no frame inicial; depois pipeline `[capture → ← → capture → ← → ...]` capturando UM print por carta destacada. Cada print classificado por `card_scan.txt` (prompt curtíssimo).
**Motivo:** carta destacada ocupa muito espaço no frame → VLM lê com alta confiança. Reduz problema de "ler 4-8 cartas pequenas e sobrepostas" a "ler 1 carta grande, N vezes".
**Trade-off:** N chamadas VLM por turno (latência ~3s × N ≈ 12-20s pra 4-7 cartas). Aceitável dentro do alvo de <15s/turno se modelo já estiver carregado.

## ADR-016: Uma ação por chamada (combate e mapa)
**Data:** 2026-05-02
**Decisão:** modelo decide só a PRÓXIMA ação. Executa, recaptura, decide de novo. Não constrói pipeline pré-computada.
**Motivo:** robusto a estado inesperado (animações, buffs/debuffs, level up no meio do combate). Erros não acumulam — auto-correção a cada step.
**Trade-off:** mais chamadas VLM por turno. Mitigado por `card_scan` cachear o conhecimento da mão (decisão usa scan + estado, não recapta tudo).

## ADR-017: Mapa via pergunta mínima de direção
**Data:** 2026-05-02
**Decisão:** prompt do `map.txt` reduz percepção a "alvo está em qual direção relativa: frente/esquerda/direita/atrás/no_alvo?". Sem contar paredes, sem ler mini-mapa em detalhe.
**Motivo:** pergunta mais fácil que VLM consegue responder. Custo de N steps por nó (4-8 × ~3s = 15-40s) é aceitável porque mapa não é caminho crítico de latência.
**Trade-off:** lento. Aceitável; mapa entre combates não bate o orçamento de <15s do combate.

## ADR-019: Consenso multi-amostra para contagem de cartas
**Data:** 2026-05-03
**Decisão:** `_perceive_combat` chama o prompt de contagem K vezes (default `PERCEPTION_COUNT_SAMPLES=3`) sobre o mesmo crop e tira o **mode** do total + idx. Pré-processamento PIL (`ImageEnhance.Contrast/Color/Sharpness`) aplicado ao crop antes da chamada. Mesmo realce reaproveitado em `scan_combat_hand` para os cards individuais.
**Motivo:** Qwen2.5-VL 7B oscila ±1 na contagem de bolinhas pequenas em leques sobrepostos; saturação/contraste destacam cyan das bolinhas; voto majoritário cancela ruído residual.
**Trade-off:** 3× chamadas VLM no estágio de contagem (~1.5s extra por turno). Vale a pena vs. erro caro de cursor (jogar carta errada). Tunável via env: `PERCEPTION_COUNT_SAMPLES=1` desativa o voto.

## ADR-018: Estados expandidos pós-jogo.md
**Data:** 2026-05-02
**Decisão:** novos estados — `CHEST_CARD_TARGET`, `BOSS_CHEST`, `STAGE_COMPLETE`, `GAME_COMPLETE` — pra cobrir fluxos descritos em jogo.md (escolha de carta-alvo do bônus, baú pós-chefe, transição entre fases, tela de vitória).
**Motivo:** sem esses, o agente trava em transições que não tinha handler.

## ADR-020: Memória injetada nos prompts de decisão
**Data:** 2026-05-03
**Decisão:** prompts `combat_decide`, `level_up` e baú recebem o `summary()` accordion + `recent(8)` da `Memory` antes do bloco de estado. Implementado em `agent._memory_block`. Após cada decisão, o handler grava o resultado em `notes.md` ("combate: jogar idx=2 (combo crescente)").
**Motivo:** sem isso, a memória escrita nunca chegava ao decisor — o agente "esquecia" a justificativa do turno anterior. Com o feedback acoplado, ele evita reescolher cartas equivalentes em level_up consecutivos e mantém continuidade de plano dentro de um combate longo.
**Trade-off:** prompts ~10-20% mais longos (latência VLM marginalmente maior). Resumo accordion é texto puro — não interfere no `response_format=json_object`. Cap de 8 eventos recentes evita explosão em runs longas.

## ADR-021: Prompt dedicado para chest_card_target
**Data:** 2026-05-03
**Decisão:** o estado `chest_card_target` (tela secundária do baú: "aplicar bônus em qual carta do deck?") usa `src/prompts/chest_card_target.txt` em vez de reusar `chest.txt`.
**Motivo:** as duas telas têm estruturas diferentes — `chest.txt` descreve a recompensa do baú (cartas/bônus/evolução/vazio), enquanto `chest_card_target` pergunta sobre cartas do deck atual com cursor visível. Reusar o prompt confundia o VLM (esperava `tipo: "carta"|"bonus"|"evolucao"|"vazio"` quando a tela era escolha de alvo).
**Trade-off:** mais um prompt para manter. Mitigado pela convenção `src/prompts/{estado}.txt` (ADR-004).
