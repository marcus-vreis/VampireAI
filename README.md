# Vampire Crawlers AI

Agente que joga **Vampire Crawlers** (deckbuilder roguelike) usando um **modelo de visão-linguagem local** (Qwen2.5-VL 7B via Ollama). Sem API paga, sem dados na nuvem.

> Status: experimental, em desenvolvimento ativo. Veja [`docs/roadmap.md`](docs/roadmap.md).

## Por quê

A maioria dos agentes LLM em jogos hoje usa modelos de fronteira (Claude, GPT, Gemini) via API. Este projeto investiga a outra ponta: **qual o menor modelo open-weight que consegue jogar um jogo turn-based razoavelmente bem**?

Vampire Crawlers foi escolhido por três razões:
- Lançou em abril/2026 — **zero contaminação** nos dados de treino dos modelos.
- Turn-based puro, sem pressão de timing.
- Decisões discretas e finitas, ideal para LLM.

## Arquitetura

```
janela do jogo
    ↓ screen capture (mss)
detector de estado (combat | map | level_up | chest | ...)
    ↓
percepção: VLM + OCR + scan sequencial de cartas
    ↓ saída validada por pydantic
estrategista LLM (decide UMA ação por chamada)
    ↓
executor → gamepad virtual (vgamepad → ViGEm Bus → DualShock 4)
    ↓
inputs no jogo (D-pad + L2/R2 + X/quadrado)
    ↺ loop
```

**Pivot 2026-05-02:** input mudou de mouse+coords para gamepad virtual. Cartas/opções são selecionadas por destaque visual ("a maior") + travessia (← →), eliminando toda calibração de UI.

Memória persistente em markdown, sumarização accordion para runs longas.

## Setup

```bash
# 1. Ollama
curl -fsSL https://ollama.com/install.sh | sh   # Linux
# Windows/macOS: https://ollama.com

ollama pull qwen2.5vl:7b
ollama serve   # deixar rodando

# 2. Projeto
git clone <repo>
cd vampire-ai
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
# Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Tesseract (para OCR auxiliar — opcional mas recomendado)
# Linux: sudo apt install tesseract-ocr tesseract-ocr-por
# Windows: baixar instalador em https://github.com/UB-Mannheim/tesseract/wiki
#          ADICIONAR ao PATH (default: C:\Program Files\Tesseract-OCR)
# macOS: brew install tesseract tesseract-lang
# Sem Tesseract o agente roda, mas perde OCR de HP/mana — VLM tem que adivinhar.

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
python -m src.perception --frame frames/example.png # debug em frame salvo
python -m src.gamepad --press confirm               # apertar 1 botão isolado
```

## Hardware mínimo

- GPU com 8GB VRAM (16GB recomendado para folga com o jogo rodando junto)
- 16GB RAM
- 10GB de disco (modelo + frames de debug)

## Limitações conhecidas

- VLM 7B alucina nomes "corrigindo" para termos formais; mitigado com prompt de transcrição literal.
- Cartas sobrepostas exigem prompt explícito de contagem.
- Latência de ~5-15s por turno; aceitável para turn-based.

## Roadmap

Veja [`docs/roadmap.md`](docs/roadmap.md). Resumo: captura → input → percepção → combate → run completa → estudo comparativo.

## Contribuição

Issues e PRs em inglês ou português, ambos OK. Antes de PR grande, abrir issue para discutir. Convenções em [`CLAUDE.md`](CLAUDE.md).

## Licença

MIT. Veja [`LICENSE`](LICENSE).

## Reconhecimentos

Inspirado por [Claude Plays Pokémon](https://www.twitch.tv/claudeplayspokemon), [PokéAgent Challenge](https://pokeagent.github.io/) e o ecossistema open source de VLMs (Qwen, Ollama).
