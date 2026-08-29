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

## ADR-022: Sensor determinístico por CV (2026-08-29) ⭐
**Data:** 2026-08-29
**Decisão:** mover percepção geométrica do VLM para OpenCV. Estado da tela,
contagem de cartas, posição do cursor, mana e navegação no mapa passam a sair de
`src/vision/`. O VLM fica com semântica: ler carta nova, escolher recompensa,
decidir a jogada.
**Motivo (medido, não suposto):**
- O recorte `hand_area` `(380,460,480,260)` cobria menos da metade do leque real
  (`~(250,380)-(1030,720)`). Num frame de 6 cartas cortava a carta da ponta
  esquerda inteira e fatiava ao meio a selecionada — que é a posição inicial do
  cursor. Como `scan_combat_hand` usava o mesmo recorte, a leitura de carta
  acontecia sobre um pedaço dela. Causa raiz da contagem errada **e** da piora
  na leitura.
- `card_scan.txt` mandava procurar o custo no "canto superior direito"; nos
  frames ele está sempre no canto superior **esquerdo**.
- Detecção de estado errava nos dois sentidos. Dos 39 frames que são o mapa, o
  VLM rotulou ao menos 9 como outra coisa — incluindo um scan de cartas de 4
  passos rodado inteiro em cima do mapa.
- Custo: `StateDetection` era 95 de 319 chamadas do log, p50 1.75s, p90 4.34s.
  A assinatura de CV equivalente roda em ~19ms.
**Trade-off aceito:** entram `numpy` e `opencv-python-headless` (~54MB), contra o
espírito de setup leve do README. Protótipo em PIL puro custava ~200ms/frame; com
numpy vai a ~5ms.
**Substitui:** ADR-009 (detecção de estado via VLM), ADR-017, ADR-019.

## ADR-023: Captura pelo client area da janela (2026-08-29)
**Decisão:** `src/window.py` localiza a janela do jogo via Win32 (`EnumWindows` +
`GetClientRect` + `ClientToScreen`) e `capture.grab` usa esse retângulo.
**Motivo:** o retângulo calculado (`monitor centralizado − 1280x720`) deixava o
frame deslocado conforme a janela tivesse barra de título ou fosse movida. Frames
de sessões diferentes têm geometrias distintas, e todo recorte de UI saía do
lugar junto. É a classe de erro que envenenava as regiões fixas com o tempo.
**Trade-off:** só Windows. Já era, por causa do ViGEm (ADR-014). Se a janela não
for achada, cai no retângulo de config com aviso.

## ADR-024: Travessia em vez de contagem num frame (2026-08-29)
**Decisão:** `scan_combat_hand` não recebe mais `total_cartas`. Percorre a mão
com ← lendo uma carta por passo e para sozinha quando o cursor volta a uma
posição já vista.
**Motivo:** a carta selecionada sobe e **cobre o círculo de custo da vizinha à
direita** — verificado em frames reais (em `card_scan_1`, Gatti Amari selecionada
esconde o custo de Pugnala). Contar círculos num único frame, portanto,
subestima em 1 sempre que o cursor não está na ponta direita. A travessia dá o
total exato e já era feita de qualquer jeito pro scan.
**Efeito colateral resolvido:** `agent.py` fixava `cursor_after_scan = 0`. Com
`total` inflado o cursor não estava em 0 e **todos** os índices seguintes saíam
deslocados — erro que acumulava, ao contrário do que a ADR-016 promete.

## ADR-025: Cache de carta por hash perceptual (2026-08-29)
**Decisão:** `src/carddb.py` guarda identidade de carta indexada por dHash 16x16
do recorte. Só há chamada ao VLM na primeira vez que uma carta aparece.
**Motivo:** uma carta é sempre o mesmo sprite; `CardScanFrame` custava 2.2s (p50)
e era a maior fatia da latência de um turno.
**Limiar medido:** sobre 256 bits. A primeira medição usou recortes do MESMO
frame deslocados de 1 a 3px (4-36 bits) — cenário fácil demais. Medindo o caso
real, a mesma carta selecionada em **frames diferentes** fica em 21-45; cartas
diferentes, em 108-134. O corte em 60 fica no meio, e erra pro lado seguro: um
par não reconhecido custa uma chamada de modelo a mais, enquanto uma fusão errada
serviria a carta errada — e a menor distância entre cartas distintas (108) está
bem acima do corte.
**Cartas buffadas geram entrada nova, e isso é correto.** Bônus alteram os números
na descrição ("Cause 374" vs "Cause 204" no mesmo Otto), então o visual muda junto
com o efeito. O cache mapeia aparência → o que a carta diz; se o que ela diz mudou,
tem que ser outra entrada.
**Não guardamos leitura incompleta:** carta com `mana=None` não entra no cache.
`combat.validate` não consegue checar mana ausente, então um custo ilegível
congelado viraria jogada ilegal em todo turno seguinte em que a carta aparecesse.

## ADR-026: Validação de jogada, não substituição do decisor (2026-08-29)
**Decisão:** o VLM continua escolhendo a carta. `src/combat.py` recusa o que o
jogo recusaria (mana insuficiente, índice fora da mão) e repergunta com o motivo;
no 3º erro joga pela regra de `jogo.md` (custo crescente, tomo mais barato
primeiro).
**Motivo:** calcular a jogada ótima por código seria mais confiável, mas tiraria
a decisão do modelo e com ela o ângulo de pesquisa do projeto. Validar torna
jogada ilegal impossível sem esvaziar o papel do VLM.
**Bônus:** a taxa de rejeição é, de graça, uma métrica de quão bom o modelo é.

## ADR-027: Resposta do VLM gravada no log (2026-08-29)
**Decisão:** `llm.py` grava o campo `response` em `logs/llm.jsonl`.
**Motivo:** antes só gravava `raw_chars`. Sem a resposta não havia como auditar
acurácia — "a leitura piorou" era impressão, não número. Sem isso o dataset
rotulado não serve pra comparar modelos.

## ADR-017 — APOSENTADA (2026-08-29)
Mapa por pergunta de direção ao VLM. Substituída pela ADR-022: o minimapa é lido
por CV e o caminho sai de um BFS. Pedir raciocínio espacial 3D a partir da visão
em 1ª pessoa era a pergunta mais difícil que dávamos a um modelo de 7B, com a
resposta desenhada ao lado na tela.

## ADR-019 — APOSENTADA (2026-08-29)
Consenso multi-amostra (k-vote) para contagem de cartas. Era curativo sobre osso
quebrado: três chamadas tirando a média de uma leitura feita num recorte errado
continuam erradas, só custam 3x mais. Com o recorte corrigido e a contagem por
travessia (ADR-024), o voto perdeu função.

## ADR-028: Ícones do minimapa por template matching (2026-08-29)
**Data:** 2026-08-29
**Decisão:** caveiras, chefe e interrogação são localizados por `cv2.matchTemplate`
contra sprites recortados de frame real (`src/vision/templates/`), não por
segmentação de cor. `src/nav.py` compõe isso com o BFS: alvo é o inimigo
alcançável mais próximo, depois o chefe, e explorar a fronteira só como pano de
fundo.
**Motivo:** segmentar por cor quase funcionou — os ícones têm um tom único
(cinza 136, contra 194-206 do piso e 160-179 da névoa). Mas as linhas de borda
das salas usam o mesmo tom e grudam no crânio, e qualquer abertura morfológica
forte o bastante pra separá-las parte o crânio ao meio. Sprite fixo pede template.
**Prioridade de alvo:** `jogo.md` diz que limpar os inimigos menores fortalece o
personagem pro chefe, então inimigo vem antes de chefe. Bônus não vale desvio.
**Escala:** varrida em passo de 0.05 entre 0.80 e 1.60, **absoluta**. Derivar do
tamanho da seta do jogador parecia natural e engana: num frame a seta mede 19px
(razão 1.19) enquanto os ícones pedem 1.30. Como é pixel art, 0.05 de erro de
escala derruba a correlação de 0.85 pra 0.60.
**Limiar por tipo:** a interrogação (template 15x9, baixo contraste) casa com
ruído do pergaminho; exige 0.90 contra 0.70 dos outros. Não é usada pra navegar.
**Validação:** no frame de referência, 6 inimigos + 1 chefe, conferidos um a um na
imagem. Num frame anterior da mesma run aparecem 7 — o inimigo a mais é o que o
jogador derrotou no intervalo.

## ADR-029: Minimapa localizado por âncora, com prova de pergaminho (2026-08-29)
**Decisão:** `minimap.locate` acha o minimapa pelo maior bloco cor-de-pergaminho
na região inferior direita, e **exige que a caixa tenha ≥50% de pergaminho**.
Devolve None quando não há mapa.
**Motivo:** a caixa fixa perdia metade do mapa em capturas desalinhadas (há
frames em que ela pega o viewport 3D no topo). Mas ancorar sem provar era pior:
num frame de combate a arte das cartas cai na mesma faixa de cinza, a busca
devolvia a área inteira, e um círculo de custo azul passava por seta do jogador.
Mapa de verdade fica em ~0.88 de pergaminho; combate, em ~0.17.

## ADR-030: Ícones fazem parte da área andável (2026-08-29)
**Decisão:** `read_minimap` soma ao piso os pixels em tom de ícone que estejam
perto do piso.
**Motivo:** caveira e chefe são desenhados em cinza 136, abaixo do limiar de piso
(185). Sem isso viram buracos na máscara e o BFS não consegue chegar até um
inimigo — que é exatamente o alvo que queremos alcançar. A exigência de
proximidade do piso exclui as bordas rasgadas do pergaminho, que usam o mesmo tom.

## ADR-031: Dois modelos, roteados pela natureza da chamada (2026-08-29)
**Data:** 2026-08-29
**Decisão:** `VLM_MODEL` atende chamadas COM imagem; `TEXT_MODEL` atende chamadas
SEM imagem. `ask_vlm` escolhe sozinho pela presença de imagem — nenhum ponto de
chamada mudou. `TEXT_MODEL` vazio cai no VLM, então instalação existente continua
funcionando.
**Motivo:** depois da ADR-022, as chamadas ao modelo se separaram em duas
naturezas bem diferentes. Transcrever carta precisa de visão. Decidir a jogada e
a recompensa é texto puro — `_decide_combat` e `_decide_choice` **já** passavam
`image=None` para um VLM. Isso desperdiça duas vezes: a torre visual não
contribui nada, e o backbone de linguagem de um VLM é mais fraco que o de um
modelo de texto do mesmo tamanho. Trocar o VLM não era a alavanca; parar de usar
VLM onde não há imagem era.
**Risco documentado:** se os dois modelos não couberem residentes na VRAM, o
Ollama descarrega e recarrega a cada troca (~30s), o que destrói o orçamento de
latência. Em 16GB: 8B + 8B é seguro, 8B + 14B é apertado. Exige
`OLLAMA_MAX_LOADED_MODELS=2`.

## ADR-032: Bench de decisão com gabarito derivado (2026-08-29)
**Data:** 2026-08-29
**Decisão:** `src/bench.py` gera cenários de combate sintéticos a partir de cartas
reais e mede três taxas por modelo: **parse** (JSON válido no schema), **legal**
(cabe na mana e no índice) e **regra** (bate com a heurística de `jogo.md`).
**Motivo:** "qual modelo usar?" precisava virar número, e a rotulagem humana para
julgar qualidade de decisão seria cara e subjetiva. Mas a legalidade de uma
jogada é uma **regra que o código já conhece** — então o benchmark corrige a
própria prova, sem rótulo nenhum. Isso o torna executável hoje, antes de existir
dataset.
**Baseline medida (qwen2.5vl:7b, seed 7):**

| n | parse | legal | regra | mediana |
|---|---|---|---|---|
| 25 (subdimensionada) | 100% | 92% | 60% | 1.62s |
| 25 (repetição) | 100% | 96% | 48% | 1.62s |
| **60** | **100±0%** | **92±7%** | **55±13%** | 1.67s |

As duas execuções de n=25 deram 60% e 48% de aderência à regra **na mesma
configuração**. Ambas estavam certas dentro do ruído: com 25 cenários a margem de
95% passa de 20pp. Ver ADR-057.

Ou seja: o modelo sempre produz JSON válido, mas **8% das jogadas são ilegais**
(o validador da ADR-026 as intercepta, ao custo de uma repergunta cada) e **40%
divergem da estratégia central do jogo**. É o número a bater ao testar um modelo
de texto dedicado.
**Ressalva (revista):** a estimativa original de "±10pp" subestimava. A margem
correta em 25 cenários é ~±20pp na taxa de aderência à regra, o que se confirmou
empiricamente: 60% e 48% na mesma configuração. Ver ADR-057.
**Limitação:** "regra" não é gabarito absoluto — uma jogada fora da heurística
pode ser melhor num contexto específico. Divergência sistemática, porém, indica
que o modelo não entendeu a mecânica de combo por custo crescente.

## ADR-033: Algarismos do HUD por glifo aprendido (2026-08-29)
**Data:** 2026-08-29
**Decisão:** `src/vision/digits.py` guarda um livro glifo→algarismo. A leitura de
mana tenta, nesta ordem: glifo conhecido (microssegundos) → Tesseract (~5ms) →
modelo (~830ms). O que Tesseract ou modelo respondem é **ensinado** ao livro.
**Motivo:** a mana é um algarismo grande e nítido sobre o orbe — trivial pra
qualquer OCR — mas o Tesseract é binário externo que nem toda instalação tem (a
desta máquina não tem), e cair no modelo custava uma chamada por turno. Como o
jogo é pixel art com fonte fixa, o glifo de um "3" é sempre o mesmo mapa de bits:
o reconhecimento é exato, não aproximado.
**Por que não embarcar templates prontos:** dos frames existentes só dá pra
colher glifos limpos de 2, 3 e 4 — os demais vieram de capturas desalinhadas e
estão cortados. Um leitor que cobre 3 dos 10 algarismos leria 0, 1, 5 e 6 errado
**em silêncio**, o que é pior que não ler. Aprender em runtime resolve sozinho.
**Proteção contra rótulo ruim:** um algarismo só passa a ser servido após 2
observações concordantes. Sem isso, uma leitura errada do modelo envenenaria
aquele glifo permanentemente.
**Mesmo padrão do `CardDB` (ADR-025):** o modelo ensina, a CV assume.

## ADR-034: Filtro de densidade separa algarismo de contorno (2026-08-29)
**Decisão:** `find_glyphs` exige densidade ≥0.30 (área do componente sobre área
da caixa), além dos filtros de tamanho e proporção.
**Motivo:** no coração, a máscara de texto claro pega também o **contorno branco
do próprio coração**, que passava nos filtros de tamanho e proporção. Medido: os
quatro algarismos ficam em 0.52-0.62 de densidade; o contorno, em 0.09-0.10 —
porque é uma curva fina ocupando uma caixa grande. Margem de 3x pros dois lados.
**Consequência:** `read_hp` passou a funcionar. A versão anterior partia a string
de dígitos ao meio e só acertava por acidente quando o total tinha contagem par.
O agrupamento agora é por **linha**, usando a altura mediana do algarismo como
corte — usar o espalhamento total colapsava as duas linhas do coração numa só.

## ADR-035: Rede de segurança contra travamento (2026-08-29)
**Data:** 2026-08-29
**Decisão:** `src/stall.py` compara a assinatura do frame entre iterações. Se a
tela não muda por `patience` passos, escalona botões — X, depois □, depois andar
pra frente. Esgotadas as tentativas, aborta a run com erro em vez de girar em
falso.
**Motivo:** o agente tem handler pra 12 estados, mas o jogo tem telas que nenhum
cobre. `jogo.md` descreve que a evolução de carta abre **duas telas de
confirmação** depois de escolher as duas cartas; nelas o `handle_chest` atual lê
`opcoes=[]` e aperta □ (sacar dinheiro) em vez de X, e o loop fica preso pra
sempre. Não temos frame de evolução pra tratar caso a caso, e nunca teremos frame
de toda tela possível.
**Por que genérico em vez de específico:** cobrir cada tela desconhecida exigiria
tê-las todas capturadas. A rede genérica custa ~1ms por passo e cobre também o
que ainda não vimos. Não substitui handler correto — evita que uma tela
desconhecida termine a run.
**Ordem dos botões:** X primeiro porque é o "confirmar/avançar" do jogo e resolve
a maioria das telas de aviso; □ depois (sacar/voltar); andar pra frente por
último, que destrava o mapa.
**Limiar medido** (diferença média por pixel numa assinatura 16x16):

| situação | delta |
|---|---|
| mesma tela, só animação de fundo | 0.89 – 1.46 |
| cursor andou uma carta (menor mudança real) | 4.44 – 8.43 |
| girou no mapa | 17.62 |

Corte em **2.5**, com ~1.7x de folga pros dois lados. O primeiro palpite foi 4.0,
que estava colado demais na menor mudança real — medir corrigiu.

## ADR-036: HP entra na decisão de combate (2026-08-29)
**Decisão:** `HandScan` carrega `hp` e o prompt de combate recebe a linha
`HP: atual/máximo`, com aviso explícito quando cai a 35% ou menos.
**Motivo:** `read_hp` funcionava mas nada consumia — era código morto, contra as
próprias regras do projeto. E sem o dado o modelo escolhe dano por padrão mesmo à
beira da morte, então as cartas de armadura do deck nunca são jogadas.
**Duas armadilhas encontradas ao implementar:**
1. O aviso usava um emoji, que estourava `UnicodeEncodeError` no console cp1252
   do Windows. Texto puro em PT-BR resolve e o modelo lê igual.
2. Numa primeira versão o HP era calculado mas **não chegava ao prompt** — a
   linha nunca foi inserida na lista de partes. Os testes exercitavam `_hp_line`
   isolado e passavam. Lição: testar a função que formata não prova que o dado
   chega ao destino. `tests/test_agent_memory.py` agora testa o prompt montado.

## ADR-037: Estado `deck` para a tela "Baralho" (2026-08-29)
**Data:** 2026-08-29
**Decisão:** 13º estado, detectado por CV. Handler fecha com quadrado.
**Motivo:** achado observando o jogo ao vivo, não nos frames salvos. As cartas do
deck também desenham círculo de custo, então a tela passava por combate e o
agente tentaria jogar carta ali — navegando o deck em vez de lutar.
**Assinatura:** painel ardósia em 0.181, entre o máximo do combate (0.052) e o
mínimo do diálogo (0.56). O alinhamento em linhas foi testado como sinal
alternativo e **não** discrimina: o leque de combate é achatado no meio e chega a
3 círculos no mesmo y.
**Ressalva:** o limiar repousa sobre **uma** observação. Confirmar com mais
amostras na sessão de rotulagem.
**Vale o registro do método:** este estado era invisível nos 118 frames salvos.
Observar o jogo rodando, mesmo sem o agente no controle, encontrou em 20 amostras
o que nenhuma análise offline encontraria.

## ADR-038: Limiar de texto do HUD adaptativo, não fixo (2026-08-29)
**Data:** 2026-08-29
**Decisão:** `hud.text_mask` calcula o limiar por Otsu sobre o canal de brilho
(restrito a pixels de baixa saturação), mais uma margem de 55% da distância até o
topo. Substitui o corte absoluto em `V > 185`.
**Motivo:** o jogo **escurece o HUD inteiro** quando um painel está aberto. Com o
corte fixo, o coração passava de 4 dígitos detectados para **zero** — descoberto
comparando o frame da tela "Baralho" com um frame normal. Otsu acha a separação
natural entre texto e fundo em qualquer iluminação.
**Por que a margem acima de Otsu:** sozinho, ele deixa o limiar baixo o bastante
pro "6" do coração grudar na barra de fração e falhar nos filtros de forma. Faixa
segura medida (4 casos, duas iluminações): 0.45 a 0.60. Escolhido 0.55.
**Efeito colateral bom:** o "4" do orbe, que produzia 2 chaves distintas de glifo
entre frames, passou a produzir **uma só** em 24 amostras ao vivo.

## ADR-039: Glifo por vizinho mais próximo (2026-08-29)
**Decisão:** `GlyphBook.lookup` procura o glifo conhecido mais parecido dentro de
72 bits, em vez de exigir chave idêntica.
**Motivo:** o mesmo algarismo não produz sempre o mesmo mapa de bits. Os dígitos
do coração medem 17px e são AMPLIADOS até a normalização 12x18, o que introduz
ruído de quantização.
**Limiar medido** sobre 216 bits: o mesmo dígito varia até 57 (o "6" é o pior
caso); dígitos diferentes ficam a 92 no mínimo. 72 fica no meio, com ~1.27x de
folga pros dois lados.

## ADR-040: Caminho que ensina os algarismos do coração (2026-08-29)
**Decisão:** `perception.read_hp_hybrid` — livro de glifos, caindo pro modelo e
ensinando o que ele responder. Simétrico ao que a mana já tinha.
**Motivo:** o HP **nunca era lido**. `read_hp` só consultava o livro, e nada
ensinava os dígitos do coração, então o dado que a ADR-036 injeta no prompt de
combate ficava sempre ausente — o recurso estava morto na prática.
**Como apareceu:** observando o jogo ao vivo. Vinte amostras mostraram `HP=None`
em todas, enquanto a mana era lida corretamente. O teste unitário não pegava
porque ele mesmo ensinava o livro antes de ler.
**Verificado ao vivo:** depois da correção, HP=(61, 61) em todas as 6 amostras
seguintes, com o modelo chamado uma vez só.

## ADR-041: Esperar o efeito, não um tempo fixo (2026-08-29)
**Data:** 2026-08-29
**Decisão:** a travessia da mão substitui o `sleep(post_dpad_settle_s)` por
captura repetida até o cursor sair do lugar, com teto de 0.8s.
**Motivo, medido antes de mexer:** cronometrando cada operação de um passo da
travessia contra o jogo aberto —

| operação | custo |
|---|---|
| captura + salvar PNG | 78 ms |
| `cv2.imread` | 13 ms |
| `detect_card_slots` | 3.3 ms |
| leitura de mana/HP em cache | ~1.6 ms |
| **`sleep` fixo pós-D-pad** | **400 ms** |

O sleep respondia por **81%** do custo. Um turno de 6 cartas jogando 4 dava ~22s,
acima da meta de 15s. Com a espera adaptativa o passo cai de ~824ms para ~424ms.
**O ganho de robustez importa mais que o de velocidade:** tempo fixo erra nos dois
sentidos — longo demais quando o jogo responde rápido, curto demais quando ele
demora, e aí a leitura acontece sobre um frame ainda em animação. Esperar o efeito
observável não tem esse problema.
**Não mexemos em `GAMEPAD.between_actions_s`** (250ms, dentro de `press`): é
timing de gamepad validado no jogo, e mudá-lo às cegas é o oposto do princípio
acima.
**Teto de 0.8s:** atingido só quando o cursor realmente não se move — na ponta do
leque, onde a travessia deve mesmo terminar.

## ADR-042: Cursor posicionado por identidade, não por índice (2026-08-29)
**Data:** 2026-08-29
**Decisão:** `perception.seek_card` anda com ←/→ até a carta em destaque **ser** a
escolhida, conferindo a identidade pelo `CardDB` a cada passo. Substitui
`select_and_confirm(alvo - cursor)`.
**Motivo — bug latente:** o agente calculava o delta, navegava e apertava X **sem
conferir onde o cursor tinha parado**. Com o índice errado por qualquer razão
(contagem, oclusão, carta comprada no meio do turno), ele jogava a carta errada em
silêncio. Não havia nenhuma verificação entre decidir e executar.
**Efeito de desempenho:** o custo de posicionar cai de uma travessia completa
(~2.5s numa mão de 6) para ~0.9s, porque a distância média até o alvo é ~1/3 da
mão. Isso **habilita** dispensar a travessia entre jogadas, mas não a dispensa
sozinho — quem faz isso é a ADR-043.

> **Correção (2026-08-29):** a versão original desta ADR e a mensagem de commit
> que a acompanhou afirmavam que a travessia entre jogadas já estava dispensada.
> Não estava: `handle_combat` continuava chamando `scan_combat_hand` a cada
> entrada. O ganho descrito era projetado, não medido no código entregue.
> Implementado em seguida na ADR-043.
**Cartas de mesmo nome são intercambiáveis:** se a mão tem dois "Tomo Vazio",
jogar qualquer um dá no mesmo, então parar no primeiro que casar é correto — e
resolve de graça a ambiguidade que uma busca por índice teria.
**Falha com segurança:** se a carta destacada não estiver no cache, ou não
pertencer à mão conhecida, devolve False sem apertar X, e o próximo passo do loop
refaz a travessia. Melhor não jogar do que jogar errado.

## ADR-043: Mão reaproveitada entre jogadas do mesmo turno (2026-08-29)
**Data:** 2026-08-29
**Decisão:** `agent._HAND` guarda a mão conhecida. Nas entradas seguintes em
combate, só mana e HP são relidos (~95ms); a lista de cartas vem do cache, menos
a que foi jogada. Zerado em toda transição de estado e sempre que o
posicionamento falha.
**Motivo:** jogar uma carta só a remove da mão — refazer a travessia inteira
depois disso custava ~2.5s numa mão de 6 e não acrescentava informação nenhuma.
**Por que é seguro:** `seek_card` (ADR-042) confere a identidade da carta antes de
confirmar. Se a mão mudou de um jeito que não prevemos — carta comprada no meio
do turno, por exemplo — ele não encontra a carta esperada, devolve False, o cache
é esquecido e o passo seguinte refaz a travessia. O cache degrada pra travessia,
nunca pra jogada errada.
**Efeito:** turno de 6 cartas jogando 4 passa de ~20s para ~13s, dentro da meta
de 15s. Combinado com a ADR-041 (espera adaptativa), o turno saiu de ~22s.
**Nota de estilo:** estado mutável em nível de módulo é feio, mas os handlers
recebem apenas `memory` e mudar essa assinatura por causa de um só handler seria
pior. `forget_hand()` torna o ciclo de vida explícito e testável.

## ADR-044: Remoção do código órfão da refatoração (2026-08-29)
**Data:** 2026-08-29
**Decisão:** apagados os schemas e prompts que a migração para CV determinística
deixou sem uso.

**Schemas removidos** (`Card`, `Enemy`, `CombatState`, `MapDirection`,
`MapAction`, `ChestAction`, `MapNode`, `MapState`): descreviam a percepção de
combate e o mapa como grafo de nós, ambos substituídos. `src/schemas.py` foi de
149 para 83 linhas.
**Mantidos** `LevelUpOption` e `ShopItem`, que parecem órfãos numa busca simples
mas são usados dentro do próprio arquivo por `LevelUpState`, `ChestState` e
`ShopState` — todos vivos.

**Prompts removidos** (`count_cards.txt`, `map.txt`, `detect_state.txt`,
`combat.txt`): a versão anterior de `docs/prompts.md` dizia que eles ficariam
"para referência histórica". Está errado — um prompt que descreve uma abordagem
aposentada engana quem abre o diretório procurando o que o agente usa hoje. O
motivo de cada aposentadoria está na ADR correspondente e o conteúdo está no git.

**`input_exec.take_money_from_chest`**: só chamava `gamepad.cancel()`, e
`handle_chest` já chama `input_exec.cancel()` direto.

**Verificação:** 121 testes passando e o replay sobre 160 frames com 0 erros
depois da remoção.

## ADR-045: Falha rápida quando o runner do modelo morre (2026-08-29)
**Data:** 2026-08-29
**Decisão:** erro de transporte cujo texto indica runner morto dispara uma sonda
de texto curta (10s). Se a sonda também falha, `ask_vlm` levanta
`ModelUnavailableError` na hora, e o agente encerra com código 3 e mensagem
acionável em vez de tratar como falha de parse.
**Motivo, observado numa execução real:** rodando o replay com o modelo, o runner
do Ollama morreu (`model runner has unexpectedly stopped`) e **toda chamada
seguinte queimava 3 tentativas de ~42s** — mais de 2 minutos por passo, com o
agente aparentando estar travado. Não havia nada no código que distinguisse "esta
chamada falhou" de "o modelo não está rodando".
**Diagnóstico descartado:** a mensagem do Ollama culpa limitação de recurso, mas
`nvidia-smi` mostrava **13.7 GB livres de 16.3 GB**. Não era VRAM. O `/api/ps`
mostrava zero modelos carregados: o runner tinha morrido e ficado morto.
**Não reproduzível sob demanda:** testando depois, imagens de seis formatos
sintéticos e seis recortes reais de carta passaram todas. Provavelmente o gatilho
foi um `GGML_ASSERT` no encoder visual sobre alguma imagem específica, que
derrubou o runner e deixou o servidor num estado ruim. A defesa vale
independentemente da causa.
**Efeito colateral bom da investigação:** os recortes de carta agora leem bem —
`Gatti Amari` mana=1, `Phiera Der Tuthello` mana=3, `Faca` mana=0. Comparado com
o cache poluído de antes (`Pughnala`, `Bastardato`, vários `mana: None`), confirma
que a correção do recorte (ADR-022) e do canto do custo no prompt surtiram efeito.

## ADR-046: Estado `notice` — saída de emergência no prompt de diálogo (2026-08-29)
**Data:** 2026-08-29
**Decisão:** `detect_dialog.txt` ganha a opção `"notice"`, para painéis que são só
texto e um botão. Handler aperta X.
**Motivo, medido:** testando o prompt contra os 12 frames de diálogo que existem
— algo que nunca tinha sido feito, porque o replay com modelo só rodou agora — ele
respondeu **"level_up" para todos os 12**, inclusive pro painel "Nenhum controle
detectado". O prompt oferecia cinco telas de recompensa e nenhuma saída, então
qualquer painel não mapeado era forçado a virar uma delas. O agente então tentava
escolher recompensa numa tela sem opção nenhuma.
**Resultado:** com a opção adicionada, **12/12 corretos** — os 11 de level up
continuam certos e o aviso sai como `notice`.
**Fecha parte de uma lacuna dada como bloqueada:** `jogo.md` descreve que a
evolução de carta abre *"uma tela, aperta X, aparece outra, aperte X novamente"*.
São exatamente painéis de aviso. Eu tinha registrado que isso precisaria de frames
reais; não precisava — precisava de uma opção honesta no prompt.
**Princípio geral:** prompt de múltipla escolha sem escape transforma "não sei" em
resposta errada com cara de certeza. Toda lista fechada de opções que vai pro
modelo merece uma saída.

## ADR-047: Telas de escolha lidas carta por carta (2026-08-29)
**Data:** 2026-08-29
**Decisão:** `perception.read_choices` detecta os círculos de custo no painel
central, recorta cada carta e lê com `card_scan.txt` — o mesmo caminho já
validado no combate. O prompt de tela inteira (`level_up.txt`, `chest.txt`) vira
reserva pras telas sem cartas.
**Motivo:** testando `level_up.txt` contra três telas reais — algo nunca feito
antes — o caminho de tela inteira errava o custo de **5 das 10** cartas, devolvia
descrições de um dígito ("1", "5"), inventava `mana=-1` e apontava a carta
selecionada errada. Mesma causa raiz do bug original de combate: a carta fica
minúscula depois do resize pra 768px.
**Resultado medido:** custo correto em 8-10 de 12, descrições reais ("Adicione 2
Mana.", "Cause 45 de dano. Chance de explodir.") e **carta selecionada 3/3**.
Os erros restantes mudam de carta entre execuções — são ruído estocástico do
modelo, não falha sistemática.

**Seleção por ALTURA, não por tamanho:** cartas de bônus trazem um orbe
decorativo maior que qualquer círculo de custo, e o critério de tamanho apontava
elas. Medido: a selecionada fica 24-30px acima das demais, e o orbe do bônus fica
no nível das não-selecionadas.

**Escala pela mediana:** usar o lado do orbe do bônus inflava o recorte dela pra
296x368 contra ~180x230 das normais, e o excesso de contexto piorava a leitura.

**Confirmação de que 4 opções existem:** um dos frames tem quatro cartas, e
`jogo.md` menciona "Aumente sua Sorte para uma chance de ter 4 escolhas". A
detecção acertou a contagem nos três.

## ADR-048: Custo de mana implausível vira `None` (2026-08-29)
**Decisão:** `CardScanFrame` valida `mana` em 0..9; fora disso, `None`.
**Motivo:** o modelo devolveu `mana=-1` lendo uma carta de bônus, onde o "-1" é o
**efeito** ("custo de mana reduzido em 1"), não o custo. Sem a validação isso
chegaria a `combat.validate`, que compara `custo > mana` — e `-1` passa em
qualquer comparação, aprovando a carta como se sempre coubesse.
**Princípio:** saída de modelo que viola uma regra conhecida do jogo deve virar
"não sei", não ser propagada. `None` já tem tratamento em todo o caminho.

## ADR-049: `deck` exige HUD presente (2026-08-29)
**Data:** 2026-08-29
**Decisão:** o veredito `DECK` passa a exigir `hud=True` além do painel ardósia.
**Motivo — regressão que eu mesmo introduzi:** o limiar da ADR-037 repousava sobre
uma única observação, e eu registrei isso como risco. O risco se materializou: o
**menu principal** fica em 0.104 de ardósia, acima do corte de 0.10, e passou a ser
classificado como baralho. O agente apertaria quadrado no menu principal.
**Por que HUD e não um limiar mais alto:** ajustar o número seria empurrar o
problema (0.104 e 0.181 não são uma separação confortável com uma amostra de cada).
O HUD é uma **regra do jogo**: só dá pra abrir o baralho dentro de uma run, e o
menu principal acontece fora. Coração e orbe presentes separam os dois por
construção, não por calibração.
**Encontrado assim:** procurando frames que caíssem em `detect_other` pra testar
aquele prompt. Deram zero — e zero era o sintoma, não o resultado.

## ADR-050: Saída de emergência também em `detect_other` (2026-08-29)
**Decisão:** `detect_other.txt` ganha `"notice"` e uma instrução explícita de não
chutar `game_over`.
**Motivo:** mesma falha de desenho da ADR-046, com consequência pior. As cinco
opções eram title / menu / game_over / stage_complete / game_complete, e
`handle_game_over` levanta `SystemExit`. Uma tela desconhecida forçada em
`game_over` **mata uma run que estava indo bem**.
**Assimetria deliberada no prompt:** errar pro lado de `notice` custa um X apertado
à toa; errar pro lado de `game_over` custa a partida. O prompt diz isso.

## ADR-051: Custo ilegível é escolha de último recurso (2026-08-29)
**Data:** 2026-08-29
**Decisão:** `fallback_index` só escolhe carta com `mana=None` quando não há
alternativa. A leitura de carta repete uma vez quando o custo sai ilegível, e o
prompt de combate avisa explicitamente sobre `"mana": null`.
**Motivo — achado rodando a cadeia completa pela primeira vez:** montei uma mão a
partir de frames reais (CV → recorte → leitura → prompt → decisão → validação) e
o modelo produziu 3 jogadas legais em 3, com raciocínio coerente. Mas uma das
cartas veio com `mana=None`, e numa das execuções o modelo escolheu justamente
ela.
**Por que isso é perigoso:** `validate` não bloqueia custo desconhecido — não dá
pra provar que a carta é cara demais. Ali sobrava mana e deu certo; com mana
apertada o jogo recusaria a jogada **em silêncio** e o turno travaria, sem nada no
log dizendo o porquê.
**Assimetria:** não bloquear é certo (não sabemos que é ilegal), mas *escolher* de
propósito é apostar. Entre uma carta que sabemos jogável e uma que não sabemos, a
certa é a conhecida.
**Não resolve o ruído do modelo:** nomes ainda saem com erro ("Pardan" por
"Pardal", "Tophello" por "Tuphello"). Isso é inofensivo — a identidade só precisa
ser consistente pra `seek_card` funcionar, e o hash do `CardDB` não depende do
texto.

## ADR-052: Agir exige o jogo em primeiro plano (2026-08-29)
**Data:** 2026-08-29
**Decisão:** `agent._require_focus` verifica `GetForegroundWindow` antes de cada
passo e levanta `GameNotFocusedError` (subclasse de `NotTheGameError`, já tratada
como fatal pelo loop). Dispensado quando o gamepad está em dry-run — no replay os
frames vêm de arquivo.
**Motivo — observado ao vivo:** durante uma sessão de observação, a mana saiu
`None, None, 24, 4, 8, 3, 3...`. Vinte e quatro de mana não existe no jogo.
Capturando e inspecionando o recorte do coração, apareceu texto da **loja da
Steam**: "TEMPO DE JOGO / nas duas semanas: 4,8 h / Total: 39,3 h". Os "24" e "8"
eram os números de tempo de jogo da Steam sendo lidos como mana.
**A causa não era o `find_game_window`:** ele achou a janela certa (uma só com
esse título, 1280x720 na posição correta). A causa é mais fundamental — **`mss`
captura uma REGIÃO DA TELA, não o conteúdo da janela**. Qualquer coisa por cima do
jogo entra no frame, e a assinatura de CV pode confundir isso com um estado real:
neste caso a página da Steam passou por `combat`.
**Por que foco é a checagem certa, e não uma assinatura mais rígida:** o gamepad
virtual só chega na janela focada. Agir sem foco seria inútil mesmo que a captura
estivesse correta. A pré-condição já era necessária; ela só não estava escrita.
**Limitação conhecida:** foco não garante ausência de sobreposição (um overlay
sempre-no-topo continua entrando). O `NOT_GAME` da ADR-022 segue como segunda
linha.

## ADR-053: Teto no resumo da memória (2026-08-29)
**Data:** 2026-08-29
**Decisão:** `Memory` limita o resumo a 24 linhas / 4000 caracteres, na escrita e
também na leitura. O corte mantém o FIM — o mais recente é o que orienta a
próxima decisão.
**Motivo — bug severo, encontrado medindo:** o `notes.md` desta máquina estava com
**254 mil caracteres**, e `agent._memory_block` injeta `summary()` inteiro em toda
decisão de combate e de escolha. Eram **~63 mil tokens por prompt**, que nenhum
contexto comporta. As decisões que testei funcionaram só porque passei
`memory=None`.
**Causa:** o accordion só comprimia quando `summarize_fn` funcionava. A versão sem
LLM fazia `prior_summary + últimos 20 eventos` — **acrescentava** a cada colapso e
nunca encolhia. Não era accordion, era acumulador. E `default_memory()` é chamado
sem `summarize_fn` no replay e em qualquer caminho que não queira gastar modelo.
**Por que cortar também na leitura:** um `notes.md` herdado de versão anterior (ou
crescido por outro caminho) não pode estourar o prompt de quem só quis ler. Defesa
nos dois lados.
**Medido depois:** o mesmo arquivo de 254 KB passa a render 2.286 caracteres
(~571 tokens) — redução de 111x. E 200 eventos sem LLM mantêm o resumo estável em
~2.200 caracteres, em vez de crescer indefinidamente.
**Lição:** o que vai dentro de um prompt precisa de teto por construção. "O
accordion cuida disso" era verdade só no caminho feliz.

## ADR-054: Memória injetada filtrada por relevância (2026-08-29)
**Data:** 2026-08-29
**Decisão:** `_memory_block` injeta só eventos de estados que orientam decisão
(combate, level up, baú, fim de fase) e descarta transições de tela. Navegação no
mapa e destravamentos deixam de ir pra memória — o `loguru` continua registrando.
**Motivo — medido, depois da ADR-053:** com o teto no resumo, o bloco caiu pra 578
tokens, mas isso ainda era **45% de um prompt de combate de 1292 tokens**. Olhando
o conteúdo: 15 eventos de navegação no mapa, 8 de destravamento, 1 de transição —
e **zero de combate**. Era histórico irrelevante disputando espaço com a mão de
cartas.
**Causa:** `handle_map` gravava um evento por passo, e cada passo dura ~1s. Em
qualquer run, a navegação afoga tudo o mais. Memória de run e diário de bordo são
coisas diferentes; estavam no mesmo lugar.
**Transições também saem:** "transição → combat" marca a estrutura da run e é útil
no arquivo, mas não diz nada sobre O QUE decidir.
**Medido:** 578 → 18 tokens no mesmo `notes.md`. Somando a ADR-053, o bloco saiu
de ~63.580 para ~18 tokens.
**O que sobra é o que importa:** "combate: jogou Otto — combo crescente",
"level up: idx=0 (sinergia com o deck)". Continuidade de plano, que era o motivo
da ADR-020 existir.

## ADR-055: Variação de grafia da ação é normalizada (2026-08-29)
**Data:** 2026-08-29
**Decisão:** `CombatAction.acao` normaliza camelCase, espaços, hífens e
maiúsculas antes de validar contra o literal.
**Motivo — achado no log:** das 15 falhas de `CombatAction` registradas, 14 foram
o incidente do runner morto (ADR-045) e **1 foi o modelo respondendo
`jogarCarta`**. A intenção estava certa e só a grafia errada, mas a validação
recusava — queimando um ciclo de repergunta (~2s) e contando como jogada ilegal na
métrica do bench.
**Limite deliberado:** normalizar grafia não é aceitar qualquer coisa. Ação
desconhecida (`descartar_mao`) continua recusada; só a forma da mesma palavra é
tolerada.

## ADR-056: Ampliar o recorte da carta NÃO melhora a leitura (resultado negativo)
**Data:** 2026-08-29
**Contexto:** o log mostra que **13% das leituras de carta voltam com mana
ilegível**. Como `VLM_IMAGE_MAX_SIDE=768` só reduz imagem e um recorte de carta
tem ~230x320, levantei a hipótese de que o dígito estava abaixo da resolução
legível.
**Testado:** as mesmas quatro cartas, com gabarito, lidas em 1x, 2x e 3x.

| escala | tamanho enviado | mana correta |
|---|---|---|
| 1x | 331x266 | 4/4 |
| 2x | 662x532 | 3/4 |
| 3x | 993x798 (reduzido a 768) | 4/4 |

**Conclusão: escala não ajuda.** O erro em 2x é ruído, não efeito de resolução — e
os nomes variando entre execuções da mesma carta ("Tumphello" / "Tuphello" /
"Tuhello", "Pardal" / "Pardalado" / "Pendrin") confirmam que a variação é
estocástica.
**Registrado como negativo de propósito:** sem isso, a hipótese seria testada de
novo. O caminho pros 13% restantes é **outro modelo**, medível com
`python -m src.bench`, não pré-processamento de imagem.
**Mitigação que fica:** a segunda leitura quando o custo sai ilegível (ADR-051).
Sendo as amostras independentes, 13% viram ~1,7%.

## ADR-057: O bench declara a própria margem de erro (2026-08-29)
**Data:** 2026-08-29
**Decisão:** o relatório imprime `taxa±margem` (95%), o default de cenários sobe
de 20 para 50, e o rodapé diz explicitamente que diferença menor que a margem não
é diferença.
**Motivo — descoberto tentando medir uma melhoria:** rerodei o bench com a mesma
seed depois de mudar o prompt de combate e normalizar a grafia da ação, pra ver
se as mudanças ajudaram. Deu **legal 92%→96%, regra 60%→48%**. Parecia que uma
métrica subiu e a outra despencou.
**Com n=60, as duas caem no meio: 92% e 55%.** As mudanças de prompt não moveram
nada mensurável — as execuções de n=25 estavam apenas oscilando.
**O achado é sobre o instrumento:** a baseline da ADR-032 foi registrada com n=25
e uma ressalva de "±10pp" que eu mesmo estimei mal. A margem real ali é ~±20pp na
aderência à regra. O número foi publicado com precisão que não tinha.

| n | margem em torno de 55% |
|---|---|
| 10 | ±31pp |
| 25 | ±20pp |
| 50 | ±14pp |
| 100 | ±10pp |

**Por que isso importa pro projeto:** a decisão de trocar de modelo repousa
inteiramente nesta métrica. Comparar dois modelos com 25 cenários cada produziria
uma "diferença" que é ruído — e a conclusão errada seria registrada como medida.
