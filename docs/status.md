# Status

## Fase atual
**Pivot arquitetural 2026-05-02 — gamepad-first.** Abandonamos mouse + coords de UI. Input agora é DualShock 4 virtual via `vgamepad` + ViGEm Bus driver.

## Decisão central
- Cartas são selecionadas por **destaque visual**, não por XY. Navegação por travessia (← →).
- Combate: scan sequencial da mão (1 print por carta), depois UMA decisão por chamada.
- Mapa: pergunta mínima de direção (frente/esquerda/direita/atrás), micro-ação por step.
- Erros não acumulam — cada passo recaptura e auto-corrige.

## Concluído

### Fundações (pré-2026-05-01)
- Setup Ollama + Qwen2.5-VL 7B
- Pipeline Python ↔ Ollama via SDK openai
- Detecção de cartas por VLM com prompt cirúrgico

### Sessão 2026-05-01 — esqueleto Fase 1 (mouse-based, depreciado)
- Estrutura inicial do repo, módulos, prompts em PT-BR, schemas, OCR auxiliar, detecção de estado por VLM
- (Mouse + coords de UI — substituído pelo gamepad)

### Sessão 2026-05-03 — contagem robusta + limpeza
- `src/perception.py` — `_count_cards` faz K-vote (default 3) sobre o mesmo crop. `enhance_for_count` aplica contraste+saturação+nitidez antes do VLM. Mesmo realce reaproveitado em `scan_combat_hand`.
- `src/config.py` — `PerceptionConfig` com `count_samples`, `enhance_contrast`, `enhance_saturation`, `enhance_sharpness`. Tudo via env `PERCEPTION_*`.
- `src/prompts/count_cards.txt` — reescrito com 3 passos explícitos (contar / conferir / identificar destacada), exemplos numéricos, instrução de "use null se houver dúvida".
- CLI: `python -m src.perception --frame F --crop hand_area --enhance` salva crop pré-processado em `frames/`.
- Cleanup `src/agent.py` — `import json` no topo (era 3× inline); removido import não usado `MapAction`, `ChestAction`.
- `tests/test_perception_vote.py` — 6 testes pra _vote_count + enhance_for_count. Suíte: 12 passing.
- ADR-019 em `docs/decisions.md`; seção "Multi-amostra para contagem" em `docs/prompts.md`.

### Sessão 2026-05-03 — memória persistente
- `src/memory.py` — implementado `Memory` com accordion: `append(event, state)`, `recent(n)`, `summary()`, `reset()`. Quando `> max_events` (default 40), colapsa eventos antigos via `summarize_fn` (LLM) e mantém os `keep_recent` (default 10) mais novos sob `## Eventos recentes`. Sem LLM, faz fallback concatenando os últimos 20 antigos.
- `src/agent.py` — wireado: registra transição de estado e falhas de percepção em `notes/notes.md`. `summarize_fn` aponta para `ask_vlm` em PT-BR.
- `tests/test_memory.py` — 5 testes (append/recent, accordion sem LLM, accordion com LLM mock, persistência roundtrip, reset). Suíte: 6 passing.

### Sessão 2026-05-03 — memória consumida + chest_card_target + housekeeping
- `src/agent.py` — `_memory_block(memory)` formata resumo accordion + últimos 8 eventos para injeção em prompts. Wireado em `_decide_combat` e `_decide_choice` (level_up, baú). Cada handler agora aceita `memory: Memory | None` e o loop passa via `handler(memory)`. Após cada decisão (combat/level_up/baú), o handler grava o resultado na memória — assim o próximo turno vê a justificativa anterior.
- `src/agent.py::handle_chest` — respeita `data["indice_selecionada"]` quando o estado é `chest_card_target` (a tela de aplicar bônus mostra o cursor sobre o deck do jogador). Fallback continua sendo `len(opcoes) - 1`.
- `src/prompts/chest_card_target.txt` — novo prompt dedicado à tela de "aplicar bônus em qual carta do deck?". Antes esse estado reusava `chest.txt` (que descreve a tela de recompensa), confundindo o VLM. Schema reusa `ChestState` mas força `tipo: "carta"`.
- `src/schemas.py::ChestState` — adicionado `indice_selecionada: int|null`.
- `src/perception.py` — `_PROMPT_BY_STATE[CHEST_CARD_TARGET]` agora aponta para `chest_card_target.txt`.
- `tests/test_input_exec.py` — 6 testes da `navigate_horizontal` + `select_and_confirm` com gamepad mockado.
- `tests/test_agent_memory.py` — 5 testes do `_memory_block` (None, vazio, recentes, cap, com resumo).
- `docs/prompts.md` — removida referência morta a `_reconcile_combat`; adicionada seção "Memória injetada nas decisões".
- `.gitignore` — adicionado (cobre `frames/`, `logs/`, `notes/`, `.env`, `.venv/`, `__pycache__/`, `image*.png` raiz, `teste.py`).
- Suíte: 23 passing.

### Sessão 2026-05-02 — pivot gamepad
- `src/gamepad.py` — wrapper vgamepad (DS4 virtual), primitivas `press`, `tap_left/right`, `confirm`, `cancel`, `turn_left/right`, `walk_*`. CLI `--test` e `--press`.
- `src/input_exec.py` — reescrito como ações de alto nível (`select_and_confirm(steps)`, `walk_forward`, etc.). Removido `click_at`, `click_card_slot`, `BUTTONS`, `HAND_AREA`.
- `src/config.py` — removido `BUTTONS`, `HAND_AREA`, `HAND_CLICK_Y_FRACTION`, `card_slot_center`, `is_coord_set`. Adicionado `GamepadConfig` com timing.
- `src/schemas.py` — adicionados `CardScanFrame`, `CombatAction`, `MapDirection`, `MapAction`, `ChoiceAction`, `ChestAction`. `Card` ganhou `selecionada`. `CombatState` ganhou `indice_selecionada`. `LevelUpOption` ganhou `e_bonus`. `ChestState` reformulado com `tipo` + `opcoes`.
- `src/states.py` — enum expandido: `CHEST_CARD_TARGET`, `BOSS_CHEST`, `STAGE_COMPLETE`, `GAME_COMPLETE`.
- `src/perception.py` — `scan_combat_hand(frame, total)` faz pipeline [print → ← → print → ←...]. Suporte aos novos estados (mapeados para mesmos prompts/schemas onde fizer sentido).
- `src/agent.py` — loop principal completo com handlers por estado. State machine, parse-fail counter, gamepad reset no finally.
- Prompts atualizados: `detect_state.txt` (12 estados), `combat.txt` (selecionada/indice), `map.txt` (pergunta de direção), `level_up.txt` (e_bonus + indice_selecionada), `chest.txt` (tipo: carta/bonus/evolucao/vazio). Novos: `card_scan.txt`, `combat_decide.txt`.
- `requirements.txt` — `vgamepad` substitui `pyautogui`.
- Docs atualizados: `states.md`, `coords.md`, `prompts.md`, `decisions.md`, `status.md`.

## Próximo
1. Instalar ViGEm Bus driver (Windows) e validar `python -m src.gamepad --test` no jogo. -> feito, funcionou
2. Validar mapeamento de "finalizar turno" — placeholder em `input_exec.end_turn()`. Pode ser Options, Triangle ou navegar até botão. -> feito
3. Confirmar comportamento do cursor entre rodadas (reseta à direita ou mantém posição). -> feito reseta a direita. mas verificar frame a frame
4. End-to-end: `--ping` → abrir jogo → `python -m src.agent --confirm --iters 5` num combate fácil. -> feito
5. ~~Injetar resumo da memória nos prompts de combate/level_up/baú~~ — feito 2026-05-03.
6. Implementar OCR de HP do jogador (já tem `read_hp_heart.txt`, falta wiring em `_perceive_combat` ou novo módulo OCR sobre `OCR_REGIONS`).
7. Validar prompt `chest_card_target.txt` em frame real (depende de capturar baú-bonus → tela secundária).
8. Definir botão real de "girar" no controle do jogo se R2/L2 não funcionar — fallback para D-pad ←/→.

## Decisões abertas
- Botão de "finalizar turno" no controle: precisa observação no jogo (ver TODO em `input_exec.py`).
- Frequência de captura: por evento (preferido) vs intervalo fixo. Hoje: cada handler captura sob demanda.
- Detecção de estado: manter só VLM ou adicionar fast-path por pixel. Custo aceito por enquanto.

## Bloqueios
- Driver ViGEm Bus precisa ser instalado manualmente no Windows (uma vez). vgamepad não instala sozinho.

## Aprendizados a preservar
- Qwen2.5-VL 7B alucina nomes "corrigindo" — prompt deve forçar transcrição literal.
- Cartas sobrepostas se perdem em prompt genérico — exigir contagem com obstruidas.
- UI em PT-BR — manter prompts em PT-BR.
- Auto-detecção de janela centralizada continua válida (vide `_primary_monitor_rect`).
- Gamepad virtual (vgamepad) entra antes do PAUSE/FAILSAFE do pyautogui — sem failsafe global agora; risco mitigado por o gamepad só afetar o jogo focado.
