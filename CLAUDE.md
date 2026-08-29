# Vampire Crawlers AI

Agente que joga Vampire Crawlers via screen capture + visão computacional + VLM
local. Open source.

## Stack
Python 3.11+, mss, vgamepad (DS4 virtual via ViGEm Bus), opencv-python-headless,
numpy, openai (apontando Ollama), pillow, pytesseract, pydantic, python-dotenv,
loguru.
Inferência local: Ollama em `http://localhost:11434/v1`. **Dois modelos**:
`VLM_MODEL` para chamadas com imagem, `TEXT_MODEL` para decisões (texto puro).
`TEXT_MODEL` vazio usa o VLM nos dois. Ver `.env.example`.
Input: gamepad virtual (D-pad + L2/R2 + X/quadrado). NÃO usa mouse — cartas
selecionadas por destaque visual, navegação por travessia.

## Divisão de trabalho: CV decide geometria, VLM decide semântica
Esta é a regra estruturante do projeto (ADR-022). Antes de mandar qualquer coisa
pro modelo, pergunte se é medível em pixels:

| Pergunta | Quem responde |
|---|---|
| Que tela é esta? | CV (`vision/screen.py`), VLM só nas raras |
| Quantas cartas? Onde está o cursor? | CV (`vision/cards.py`) |
| Quanta mana? | glifo aprendido → OCR → modelo (`vision/digits.py`) |
| Pra onde ando? | BFS no minimapa + ícones (`src/nav.py`) |
| Que carta é esta? | `CardDB`; VLM só na 1ª aparição |
| Qual jogar? Qual recompensa pegar? | `TEXT_MODEL`, validado por código |

Se você está prestes a pedir ao VLM uma contagem, uma posição ou uma direção,
pare: isso é bug de arquitetura, não prompt mal escrito.

## Layout
- `src/window.py` — client area da janela via Win32
- `src/capture.py` — screenshot dessa área
- `src/vision/` — CV pura, funções sobre arrays numpy, sem efeito colateral
  - `cards.py` — círculos de custo → contagem, cursor, bbox da carta
  - `screen.py` — classificação de tela por assinatura de cor
  - `minimap.py` — posição, direção, BFS de navegação
  - `icons.py` — ícones do minimapa por template matching
  - `hud.py` — mana e HP do HUD
  - `digits.py` — livro glifo→algarismo, aprendido em runtime
  - `regions.py` — regiões em fração do client area
  - `debug.py` — anota um frame com tudo que a CV vê
  - `templates/` — sprites de ícone recortados de frame real
- `src/gamepad.py` — primitivas de gamepad
- `src/input_exec.py` — ações de alto nível
- `src/perception.py` — orquestra captura + gamepad + VLM (travessia da mão)
- `src/carddb.py` — cache de carta por hash perceptual
- `src/nav.py` — escolha de alvo no mapa (inimigo → chefe → explorar)
- `src/combat.py` — validação de jogada e regra de reserva
- `src/stall.py` — detecta tela travada e escalona botões pra destravar
- `src/agent.py` — loop principal, state machine
- `src/label.py` — sessão de captura rotulada → `dataset/`
- `src/bench.py` — compara modelos na decisão de combate, gabarito derivado
- `src/memory.py` — notes.md persistente, sumarização accordion
- `src/llm.py` — wrapper Ollama via SDK openai
- `src/schemas.py` — pydantic models
- `src/config.py` — paths, janela, GAMEPAD timing, constantes
- `src/states.py` — enum e detector híbrido de estados
- `src/prompts/` — templates por estado, em PT-BR
- `dataset/` — frames rotulados, **versionado** (é a suíte de regressão)
- `frames/` — despejo de debug, gitignored
- `logs/`, `notes/` — gitignored
- `docs/` — status, coords, states, prompts, decisões
- `tests/` — pytest

## Comandos
- `ollama serve` — sobe servidor (terminal separado)
- `python -m src.llm --ping` — testa Ollama responde
- `python -m src.window` — confere que a janela do jogo foi localizada
- `python -m src.capture --once` — 1 screenshot pra debug
- `python -m src.states --frame F.png --cv-only` — assinatura de CV, sem VLM
- `python -m src.perception --cards F.png` — detecção de cartas num frame
- `python -m src.vision.debug F.png` — anota o frame com tudo que a CV vê
- `python -m src.gamepad --test` — sequência de teste do gamepad
- `python -m src.input_exec --action confirm` — testa uma ação
- `python -m src.label` — sessão de rotulagem (interativo)
- `python -m src.label --summary` — concordância CV × rótulo por estado
- `python -m src.bench --models a,b` — compara modelos na decisão (consome GPU)
- `python -m src.agent` — loop completo (consome GPU; **pedir confirmação**)
- `pytest -q` — roda testes
- `ruff check . && ruff format .` — lint + format

## Convenções de código
- Type hints em toda função pública
- Funções <30 linhas; quebrar se passar
- `loguru` para logs; nunca `print` (exceto saída de CLI)
- Magic numbers e timing em `config.py`; limiares de CV como constantes de módulo
  no arquivo que os usa, **sempre com o valor medido no comentário**
- Saídas de LLM validadas via pydantic em `schemas.py`
- Toda chamada LLM com retry+backoff e log estruturado em `logs/llm.jsonl`,
  incluindo a resposta (sem ela não há como medir acurácia)
- `src/vision/` é puro: recebe array, devolve dado. Captura e gamepad ficam fora
- Imports ordenados: stdlib, third-party, local
- Docstrings só em funções públicas não-triviais; nunca docstring óbvia
- Nomes de funções em inglês; strings de prompt e docstrings em PT-BR

## Regras
- Nunca rodar `agent.py` sem confirmar (consome GPU pesado)
- **Antes de afinar um detector de CV, olhar a máscara.** Iterar em limiar no
  escuro custa mais que renderizar o overlay e ver o que está sendo pego
- **Medir antes de decidir.** Todo limiar deste projeto tem número que o
  justifica; ao mudar um, registrar a nova medição
- **Testar o caminho, não só a peça.** Um `_hp_line` correto com teste passando
  não provou que o HP chegava ao prompt — não chegava. Prompts e payloads que
  vão pro modelo merecem teste do texto montado
- **Observar o jogo rodando acha o que frame salvo não acha.** O estado `deck`, o
  escurecimento do HUD e o HP nunca lido saíram todos de
  `python -m src.label --watch`, que não envia input nenhum
- **Nada de caractere fora do cp1252 em texto que possa ser impresso.** Seta e
  emoji estouram `UnicodeEncodeError` no console do Windows; já aconteceu 3x
- Nunca commitar `frames/`, `logs/`, `notes/`, `.env`, modelos. `dataset/` SIM
- Nunca tocar arquivos do Steam ou da pasta do jogo
- Antes de mudar prompts, ler `@docs/prompts.md`
- Antes de mudar regiões ou limiares de cor, ler `@docs/coords.md`
- Status, fase, TODOs em `@docs/status.md` — atualizar ao concluir etapa
- Se o VLM falhar parsing 3x seguidas, abortar turno e logar — não chutar
- Tela desconhecida não pode matar a run: o antitravamento (`src/stall.py`) é a
  rede genérica. Handler específico é melhor, mas só quando houver frame real
- Prompts ao VLM exigem transcrição literal, sem correção ortográfica
- **Toda lista fechada de opções num prompt precisa de saída.** Sem ela, "não sei"
  vira resposta errada com cara de certeza: `detect_dialog` sem a opção `notice`
  respondia "level_up" para um painel de aviso

## Jogo
Vampire Crawlers, deckbuilder turn-based. UI em **português**, jogado com
controle. Janela windowed 1280x720. 12 estados. Detalhes em `@docs/states.md`,
convenções de prompt em `@docs/prompts.md`, mecânicas em `@jogo.md`.

Fatos do jogo que moldam o código e não são óbvios:
- O círculo de custo da carta **selecionada pulsa entre azul e magenta**, então
  detecção por cor precisa cobrir as duas faixas; o cursor é achado por tamanho.
- A carta selecionada sobe e **cobre o círculo de custo da vizinha à direita**.
  Por isso a contagem num único frame subestima, e o scan usa travessia.
- No minimapa, ícones (caveira, chefe, bônus) são cinza 136, **abaixo** do limiar
  de piso. Precisam ser somados à área andável, senão viram buracos e o BFS não
  alcança o inimigo.
- Bônus e baús encostados em parede são desenhados deslocados pra **borda** da
  célula, não no centro dela.

## Performance
Latência alvo por turno: <15s. Depois da ADR-022 o orçamento típico de um turno
de combate é 1-2 chamadas ao VLM (era 6+N). Modelo carrega em ~30s na 1ª chamada
— manter Ollama rodando entre sessões.

Custos medidos contra o jogo aberto (mediana de 5): captura+PNG 78ms, `imread`
13ms, `detect_card_slots` 3.3ms, mana/HP em cache ~1.6ms. **A CV não é o
gargalo** — espera por input é. Por isso a travessia espera o cursor mover em vez
de dormir um tempo fixo (ADR-041).

A leitura de mana se auto-otimiza: cada algarismo custa uma chamada de modelo até
aparecer duas vezes, e depois é reconhecido localmente (ADR-033). Instalar o
Tesseract acelera esse aquecimento, mas não é mais necessário.

## Open source
Licença MIT. README com setup, demo gif, contribuição. Sem segredos no histórico
git. Issues e PRs em inglês ou português, ambos OK.
