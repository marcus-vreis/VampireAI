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
**Baseline medida (qwen2.5vl:7b, 25 cenários, seed 7):**

| parse | legal | regra | mediana |
|---|---|---|---|
| 100% | 92% | 60% | 1.62s |

Ou seja: o modelo sempre produz JSON válido, mas **8% das jogadas são ilegais**
(o validador da ADR-026 as intercepta, ao custo de uma repergunta cada) e **40%
divergem da estratégia central do jogo**. É o número a bater ao testar um modelo
de texto dedicado.
**Ressalva:** com n=25 a margem ainda é larga (~±10pp em 92%). Uma amostra de 5
cenários dava 80%/20%, o que mostra que amostra pequena aqui é ruído, não sinal.
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
