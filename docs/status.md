# Status

## Fase atual
**Sensor determinístico, 2026-08-29.** A percepção geométrica saiu do VLM e foi
para OpenCV (`src/vision/`). O modelo ficou com semântica: ler carta nova,
escolher recompensa, decidir a jogada — esta última validada por código.

Meta corrente: **zerar a fase 1**.

## Decisão central
- Estado da tela, contagem de cartas, cursor e navegação: CV, ~19ms, sem alucinar.
- Combate: travessia da mão (não precisa saber o total antes) + cache de carta por
  hash + jogada do VLM validada contra mana e índices.
- Mapa: BFS sobre o minimapa. Sem chamada de modelo.
- Erros não acumulam — cada passo recaptura.

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

### Sessão 2026-08-29 — sensor determinístico (ADR-022 a ADR-027)

**Diagnóstico com evidência, antes de mexer em qualquer código:**
- `hand_area` `(380,460,480,260)` cobria menos da metade do leque real. Num frame
  de 6 cartas cortava a carta da ponta esquerda inteira e fatiava ao meio a
  selecionada — a posição inicial do cursor. `scan_combat_hand` usava o mesmo
  recorte, então a leitura de carta acontecia sobre um pedaço dela. **Causa raiz
  da contagem errada e da piora na leitura.**
- `card_scan.txt` pedia o custo no "canto superior direito"; ele está sempre no
  **esquerdo**.
- Detecção de estado errava nos dois sentidos. De 39 frames que são o mapa, ao
  menos 9 foram rotulados como outra coisa — incluindo um scan de cartas de 4
  passos rodado inteiro em cima do mapa.
- `agent.py` fixava `cursor_after_scan = 0`: com o total inflado, todo índice
  seguinte saía deslocado. Erro que acumulava.
- `llm.py` gravava só `raw_chars`, não a resposta — não havia como medir nada.
- Frames de sessões diferentes tinham geometrias distintas: a captura era um
  retângulo calculado, não a janela real.

**Módulos novos:**
- `src/window.py` — client area via Win32, com fallback pro retângulo de config.
- `src/vision/cards.py` — círculos de custo → contagem, cursor, bbox da carta.
  Cobre azul e magenta (o círculo da selecionada **pulsa** entre os dois) e
  descarta texto azul da descrição exigindo núcleo claro (o algarismo).
- `src/vision/screen.py` — classificação de tela por assinatura de cor.
- `src/vision/minimap.py` — posição, direção e BFS até a fronteira inexplorada.
- `src/vision/hud.py` — mana e HP por OCR, com fallback pro VLM.
- `src/vision/regions.py` — regiões em fração do client area.
- `src/carddb.py` — cache de carta por dHash. Limiar 60/256 bits, medido:
  mesma carta deslocada 1-3px dá 4-36; cartas diferentes, 112-131.
- `src/combat.py` — validação de jogada e regra de reserva.
- `src/label.py` — sessão de captura rotulada → `dataset/`.

**Reescritos:** `perception.py` (travessia + cache), `states.py` (híbrido
CV/VLM), `agent.py` (handlers usando CV, laço de validação da jogada),
`capture.py` (janela real).

**Prompts:** `detect_dialog.txt` e `detect_other.txt` novos (perguntas restritas
ao subgrupo que a CV já isolou). `card_scan.txt` corrigido.

**Medido:** classificação de tela 19ms contra ~1750ms (p50) do VLM. Zero erro em
mapa/combate/diálogo nos 118 frames existentes; os 4 restantes são 2 falhas reais
de captura (janela errada), 1 menu roteado ao VLM e 1 frame de layout antigo.

**Testes:** 61 passando. `tests/test_vision.py`, `test_combat_rules.py`,
`test_carddb.py` novos; `test_perception_vote.py` removido (testava o k-vote da
ADR-019, aposentada).


### Sessão 2026-08-29 (cont.) — navegação com alvo (ADR-028 a ADR-030)

O mapa passou de "explorar fronteira" para **mirar inimigo, depois chefe**, que é
o que a run precisa pra chegar ao fim da fase.

- `src/vision/icons.py` — caveira, chefe e `?` por template matching contra
  sprites recortados de frame real (`src/vision/templates/`). Segmentar por cor
  quase deu certo (os ícones têm tom único, cinza 136), mas as linhas de borda das
  salas usam o mesmo tom e grudam no crânio; abertura morfológica forte o bastante
  pra separá-las parte o crânio ao meio.
- `src/nav.py` — prioridade de alvo: inimigo alcançável mais próximo → chefe →
  fronteira. Segue `jogo.md`: limpar os menores fortalece pro chefe.
- `src/vision/debug.py` — anota um frame com tudo que a CV vê. Nasceu da lição
  mais cara desta sessão: olhar a máscara resolve em uma tentativa o que rodadas
  de ajuste de limiar às cegas não resolvem.
- `minimap.locate` — acha o minimapa por âncora de pergaminho **e exige ≥50% de
  pergaminho na caixa**. Sem essa prova, num frame de combate a arte das cartas
  passava por mapa e um círculo de custo azul virava seta do jogador (bug real,
  encontrado por teste).
- `minimap.read_minimap` — ícones entram na máscara andável. Eram cinza 136,
  abaixo do limiar de piso, então viravam buracos e o BFS não chegava no inimigo.

**Medido:** no frame de referência, 6 inimigos + 1 chefe, conferidos um a um na
imagem. Num frame anterior da mesma run aparecem 7 — o a mais é o inimigo que o
jogador derrotou no intervalo, o que valida a detecção contra a linha do tempo.
Zero minimapa falso em combate nos 118 frames. `find_icons` custa ~55ms; a mesma
pergunta ao VLM custava ~2.4s.

**Escala de template:** varrida em passo de 0.05 entre 0.80 e 1.60, absoluta.
Derivar do tamanho da seta engana — num frame a seta é 19px (razão 1.19) e os
ícones pedem 1.30.

**Testes:** 69 passando (`tests/test_nav.py` novo, 8 testes).


### Sessão 2026-08-29 (cont. 2) — dois modelos + baseline medida (ADR-031, ADR-032)

- `src/config.py::LLMConfig` — `vision_model` e `text_model`, com `pick(has_image)`.
  `TEXT_MODEL` vazio cai no VLM: instalação existente não muda.
- `src/llm.py` — roteamento automático pela presença de imagem; `model=` força um
  específico (usado pelo bench). `--ping` testa os dois e avisa quando `TEXT_MODEL`
  não está configurado.
- `src/bench.py` — comparação de modelos na decisão de combate, com gabarito
  **derivado**: a legalidade de uma jogada é regra que o código já conhece, então
  não precisa de rotulagem humana e roda hoje.
- `.env.example` — documenta a divisão dos dois modelos e o risco de VRAM.
- **Limpeza de config morta:** `PerceptionConfig` (tunable do k-vote, ADR-019
  aposentada), `UI_REGIONS`, `OCR_REGIONS`, `OCR_UPSCALE`, `OCR_THRESHOLD`,
  `is_region_set` — nada disso tinha uso restante. Importava remover: o
  `UI_REGIONS["hand_area"] = (380, 460, 480, 260)` era exatamente a região
  quebrada que causou o bug original, à mão pra ser reusada por engano.

**Baseline do modelo atual** (`qwen2.5vl:7b`, 25 cenários, seed 7):

| parse | legal | regra | mediana |
|---|---|---|---|
| 100% | 92% | 60% | 1.62s |

8% das jogadas são ilegais (o validador intercepta, ao custo de uma repergunta) e
40% divergem da estratégia de custo crescente. É o número a bater.

**Testes:** 74 passando (`tests/test_bench.py` novo, 5 testes).


### Sessão 2026-08-29 (cont. 3) — algarismos aprendidos (ADR-033, ADR-034)

Última chamada de modelo por turno eliminada, e sem depender do Tesseract.

- `src/vision/digits.py` — livro glifo→algarismo com votos, persistido em
  `notes/glyphs.json`. Ordem de leitura: glifo conhecido → Tesseract → modelo.
  O que Tesseract ou modelo respondem é ensinado ao livro, então a chamada some
  depois que cada algarismo aparece duas vezes. Mesmo padrão do `CardDB`.
- **Por que não embarcar templates prontos:** dos frames existentes só dá pra
  colher glifos limpos de 2, 3 e 4 — os demais vieram de capturas desalinhadas e
  estão cortados. Um leitor cobrindo 3 dos 10 algarismos leria 0, 1, 5 e 6 errado
  em silêncio, o que é pior que não ler.
- `find_glyphs` ganhou filtro de **densidade**: no coração, a máscara de texto
  claro pegava o contorno branco do próprio coração, que passava nos filtros de
  tamanho e proporção. Medido: algarismos em 0.52-0.62, contorno em 0.09-0.10.
- `read_hp` passou a funcionar de verdade: agrupa glifos por **linha** usando a
  altura mediana como corte. A versão anterior partia a string de dígitos ao meio
  e só acertava por acidente com contagem par.

**Verificado:** no frame de referência o orbe dá 1 glifo e o coração dá 4 em 2
linhas (61/61). O portão de votos funciona: desconhecido → None, 1 voto → None,
2 votos → lê.

**Testes:** 83 passando (`tests/test_digits.py` novo, 9 testes).


### Sessão 2026-08-29 (cont. 4) — antitravamento e HP na decisão (ADR-035, ADR-036)

Levantamento do que ainda bloqueia zerar a fase 1 apontou o mesmo padrão nos dois
buracos achados: **o agente trava numa tela que nenhum handler cobre.**

`jogo.md` descreve que a evolução de carta abre duas telas de confirmação depois
de escolher as duas cartas. Nelas o `handle_chest` lê `opcoes=[]` e aperta □ em
vez de X — preso pra sempre. Não temos frame de evolução, e nunca teremos frame
de toda tela possível.

- `src/stall.py` — compara assinatura 16x16 entre iterações; tela parada por N
  passos dispara escalonamento X → □ → frente, e aborta se nada destravar.
  Ligado em `agent._step`, antes da percepção (não adianta perceber de novo uma
  tela que não mudou).
- **Limiar medido**, não chutado: tela parada com só animação de fundo fica em
  0.89-1.46; a menor mudança real que precisamos ver (cursor andando uma carta)
  é 4.44; girar no mapa, 17.62. Corte em 2.5. O primeiro palpite foi 4.0, colado
  demais na menor mudança real.
- `HandScan.hp` + `agent._hp_line` — HP entra no prompt de combate, com aviso
  quando cai a 35% ou menos. `read_hp` existia mas nada consumia; sem o dado o
  modelo escolhe dano por padrão mesmo à beira da morte e as cartas de armadura
  nunca são jogadas.

**Testes:** 93 passando (`tests/test_stall.py` novo, 7 testes; 3 novos para HP).


## Próximo

1. **Sessão de rotulagem** — `python -m src.label --details`. Alvo: 60-100 frames
   cobrindo baú, chefe, game over, shop e as fases 2+. É o que transforma
   "parece melhor" em número e o que a comparação entre modelos vai usar.
2. **Ligar a suíte de regressão ao dataset** — hoje `tests/test_vision.py` usa
   frames rotulados à mão de `frames/`, que é gitignored. Migrar pra `dataset/`.
3. **Rodar no jogo** — `python -m src.agent --confirm --iters 10` num combate
   fácil. Nada do sensor novo foi exercitado contra o jogo rodando ainda; toda
   validação até agora é sobre frames salvos.
4. **Testar um modelo de texto no `TEXT_MODEL`.** A infraestrutura está pronta
   (ADR-031) e a baseline está medida (ADR-032). Baixar um candidato e rodar
   `python -m src.bench --models qwen2.5vl:7b,candidato --scenarios 40`. Meta:
   subir "regra" acima de 60% sem perder "legal".
5. **Baú e pedra no minimapa** — faltam templates para baú e obstáculo. Sem o de
   pedra, o BFS pode tentar atravessar bloqueio; sem o de baú, o agente ignora
   recompensa no caminho. Precisa de frames que os contenham.
6. **Features de aresta no mapa** — bônus e baús encostados em parede são
   desenhados deslocados pra borda da célula (visível nos losangos ao lado das
   caveiras). Hoje eles entram na área andável mas não são alvos. A estrutura
   certa é grade de células + features presas a arestas.
7. **Popular o livro de glifos** — na primeira run, cada algarismo custa uma
   chamada de modelo até aparecer duas vezes. Depois disso a leitura é local.
   Instalar o Tesseract acelera esse aquecimento, mas não é mais necessário.
8. **Fluxo de evolução de carta** — o antitravamento (ADR-035) evita que a run
   morra ali, mas o tratamento correto (escolher duas cartas, confirmar duas
   vezes) precisa de frames reais pra ser escrito. Capturar durante a rotulagem.
9. **Fim de fase** — depois do chefe morto, `nav.plan` não acha inimigo nem chefe
   e cai em explorar fronteira. Deveria procurar a saída (pá no chão / quadrado
   preto com o ícone do personagem). Também precisa de frames.

## Decisões abertas
- Trocar o VLM de visão: medir no dataset antes de baixar modelo.
- Botão de "finalizar turno": validado como Bola/Circle, confirmar em fase 2+.
- `dataset/` versionado (é a suíte de regressão) vs. peso no repo.

## Bloqueios
- Driver ViGEm Bus precisa ser instalado manualmente no Windows (uma vez). vgamepad não instala sozinho.

## Aprendizados a preservar
- Qwen2.5-VL 7B alucina nomes "corrigindo" — prompt deve forçar transcrição literal.
- Cartas sobrepostas se perdem em prompt genérico — exigir contagem com obstruidas.
- UI em PT-BR — manter prompts em PT-BR.
- Auto-detecção de janela centralizada continua válida (vide `_primary_monitor_rect`).
- Gamepad virtual (vgamepad) entra antes do PAUSE/FAILSAFE do pyautogui — sem failsafe global agora; risco mitigado por o gamepad só afetar o jogo focado.
