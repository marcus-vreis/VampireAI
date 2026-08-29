# Vampire Crawlers AI

Agente que joga **Vampire Crawlers** (deckbuilder roguelike) com **visão computacional** para perceber a tela e um **modelo local** (via Ollama) para decidir. Sem API paga, sem dados na nuvem.

> Status: experimental, em desenvolvimento ativo. Veja [`docs/roadmap.md`](docs/roadmap.md).

## Por quê

A maioria dos agentes LLM em jogos hoje usa modelos de fronteira (Claude, GPT, Gemini) via API. Este projeto investiga a outra ponta: **qual o menor modelo open-weight que consegue jogar um jogo turn-based razoavelmente bem**?

Vampire Crawlers foi escolhido por três razões:
- Lançou em abril/2026 — **zero contaminação** nos dados de treino dos modelos.
- Turn-based puro, sem pressão de timing.
- Decisões discretas e finitas, ideal para LLM.

## Arquitetura

O princípio estruturante: **a CV decide geometria, o modelo decide semântica.**

```
janela do jogo
    ↓ captura do client area (Win32 + mss)
CV determinística (~19ms, sem alucinar)
    ├─ que tela é esta?        assinatura de cor
    ├─ quantas cartas? cursor? círculos de custo
    ├─ quanta mana?            OCR
    └─ pra onde ando?          minimapa → ícones → BFS
    ↓
modelo, só onde há semântica
    ├─ que carta é esta?       VLM_MODEL (1ª vez; depois é cache por hash)
    └─ qual jogar? qual pegar? TEXT_MODEL (texto puro, sem imagem)
    ↓ validado por código: mana e índice, senão repergunta
executor → gamepad virtual (vgamepad → ViGEm Bus → DualShock 4)
    ↺ recaptura a cada passo, erros não acumulam
```

Isso não era assim. A percepção inteira passava pelo VLM, e medir mostrou o custo:
de 39 frames que eram o mapa, o modelo rotulava ao menos 9 como outra coisa — a
ponto de rodar um scan de cartas de 4 passos em cima do mapa. Perguntar "quantas
cartas?" e "pra onde ando?" a um modelo de 7B era erro de arquitetura, não prompt
mal escrito. Ver [`docs/decisions.md`](docs/decisions.md), ADR-022.

O modelo continua decidindo o que importa — e a taxa de jogadas ilegais que o
validador intercepta virou métrica de qualidade dele.

**Input por gamepad virtual.** Cartas e opções são escolhidas por destaque visual
("a maior") + travessia (← →), sem nenhuma coordenada de clique.

Memória persistente em markdown, sumarização accordion para runs longas.

## Ferramentas

```bash
python -m src.vision.debug frames/x.png    # anota o frame com tudo que a CV vê
python -m src.label                        # sessão de captura rotulada → dataset/
python -m src.label --summary              # concordância CV × rótulo, por estado
python -m src.bench --models a,b           # compara modelos na decisão de combate
```

O `bench` corrige a própria prova: a legalidade de uma jogada é regra que o código
já conhece, então dá pra comparar modelos sem rotulagem humana.

## Setup

```bash
# 1. Ollama
curl -fsSL https://ollama.com/install.sh | sh   # Linux
# Windows/macOS: https://ollama.com

ollama pull qwen2.5vl:7b
ollama serve   # deixar rodando

# Opcional: um modelo de TEXTO separado para as decisões. As chamadas de decisão
# não têm imagem, então um VLM ali é desperdício. Configure TEXT_MODEL no .env.
# Cuidado com VRAM: os dois precisam caber residentes, senão o Ollama recarrega
# a cada troca (~30s). Ver .env.example.

# 2. Projeto
git clone <repo>
cd vampire-ai        # os comandos abaixo precisam rodar DAQUI
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
# Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# As dependências são declaradas em pyproject.toml; requirements.txt aponta pra
# lá, então não há duas listas pra manter em sincronia.
# O agente é uma aplicação, roda com `python -m src.agent` de dentro do projeto.

# 3. Tesseract (para OCR auxiliar — opcional mas recomendado)
# Linux: sudo apt install tesseract-ocr tesseract-ocr-por
# Windows: baixar instalador em https://github.com/UB-Mannheim/tesseract/wiki
#          ADICIONAR ao PATH (default: C:\Program Files\Tesseract-OCR)
# macOS: brew install tesseract tesseract-lang
# Sem Tesseract o agente roda, mas a leitura de mana volta a custar uma chamada
# ao modelo por turno em vez de ~5ms.

# 4. ViGEm Bus driver (Windows) — necessário pro gamepad virtual
# https://github.com/ViGEm/ViGEmBus/releases  (instalar uma vez)

# 5. Vampire Crawlers
# Compre na Steam. Janela windowed 1280x720; o agente auto-detecta o centro do monitor primário.

# 6. Sanidade
python -m src.llm --ping
python -m src.capture --once
python -m src.gamepad --test    # foque o jogo: deve ver botões sendo apertados
```

## Uso

```bash
python -m src.agent --confirm                       # loop completo (consome GPU)
python -m src.agent --confirm --iters 10            # bounded para teste
python -m src.perception --cards frames/x.png       # detecção de cartas, sem modelo
python -m src.states --frame frames/x.png --cv-only # assinatura de CV, sem modelo
python -m src.gamepad --press confirm               # apertar 1 botão isolado
```

## Hardware mínimo

- GPU com 8GB VRAM (16GB recomendado — dá folga pro jogo e permite um segundo
  modelo de texto residente)
- 16GB RAM
- 10GB de disco (modelo + frames de debug)

## Limitações conhecidas

- VLM 7B alucina nomes "corrigindo" para termos formais; mitigado com prompt de
  transcrição literal.
- Na decisão de combate, o `qwen2.5vl:7b` acerta 92% em legalidade e 60% em
  aderência à estratégia do jogo (25 cenários). O validador intercepta o ilegal,
  mas cada interceptação custa uma repergunta. Um `TEXT_MODEL` dedicado deve
  melhorar isso — é o próximo experimento.
- Faltam templates de baú e obstáculo no minimapa; sem o de obstáculo, o
  planejador pode tentar rota bloqueada.
- Windows apenas, por causa do driver ViGEm.

## Roadmap

Veja [`docs/roadmap.md`](docs/roadmap.md). Resumo: captura → input → percepção → combate → run completa → estudo comparativo.

## Contribuição

Issues e PRs em inglês ou português, ambos OK. Antes de PR grande, abrir issue para discutir. Convenções em [`CLAUDE.md`](CLAUDE.md).

## Licença

MIT. Veja [`LICENSE`](LICENSE).

## Reconhecimentos

Inspirado por [Claude Plays Pokémon](https://www.twitch.tv/claudeplayspokemon), [PokéAgent Challenge](https://pokeagent.github.io/) e o ecossistema open source de VLMs (Qwen, Ollama).
