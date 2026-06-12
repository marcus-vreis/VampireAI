# Vampire Crawlers AI

Agente que joga Vampire Crawlers via screen capture + VLM local. Open source.

## Stack
Python 3.11+, mss, vgamepad (DS4 virtual via ViGEm Bus), openai (apontando Ollama), pillow, pytesseract, pydantic, python-dotenv, loguru.
Inferência local: Ollama em `http://localhost:11434/v1`, modelo `qwen2.5vl:7b`.
Input: gamepad virtual (D-pad + L2/R2 + X/quadrado). NÃO usa mouse — cartas selecionadas por destaque visual, navegação por travessia.

## Layout
- `src/capture.py` — screenshot da janela
- `src/gamepad.py` — primitivas de gamepad (press, tap_left/right, confirm, cancel, turn_*)
- `src/input_exec.py` — ações de alto nível (select_and_confirm, walk_forward, etc.)
- `src/perception.py` — VLM + OCR auxiliar + scan sequencial de cartas em combate
- `src/agent.py` — loop principal, state machine com handlers por estado
- `src/memory.py` — notes.md persistente, sumarização accordion (placeholder)
- `src/llm.py` — wrapper Ollama via SDK openai
- `src/schemas.py` — pydantic models de percepção e de ações do agente
- `src/config.py` — paths, janela, GAMEPAD timing, OCR_REGIONS, constantes
- `src/states.py` — detector e enum de estados (12 estados)
- `src/prompts/` — templates de prompt por estado, em PT-BR
- `frames/` — debug, gitignored
- `logs/` — jsonl de chamadas e raciocínio, gitignored
- `notes/` — memória persistente do agente, gitignored
- `docs/` — status, coords, roadmap, decisões
- `tests/` — pytest, smoke + unit

## Comandos
- `ollama serve` — sobe servidor (terminal separado)
- `ollama ps` — confere modelo carregado
- `python -m src.llm --ping` — testa Ollama responde
- `python -m src.capture --once` — 1 screenshot pra debug
- `python -m src.gamepad --test` — sequência fixa de teste do gamepad virtual
- `python -m src.input_exec --action confirm` — testa uma ação de alto nível
- `python -m src.perception --frame frames/x.png` — testa percepção em frame salvo
- `python -m src.agent` — loop completo (consome GPU; pedir confirmação)
- `pytest -q` — roda testes
- `ruff check . && ruff format .` — lint + format

## Convenções de código
- Type hints em toda função pública
- Funções <30 linhas; quebrar se passar
- `loguru` para logs; nunca `print`
- Magic numbers e timing só em `config.py`, nunca inline. Sem coords de UI (input é gamepad)
- Saídas de LLM validadas via pydantic em `schemas.py`
- Toda chamada LLM com retry+backoff e log estruturado em `logs/llm.jsonl`
- Frames como `frames/{ISO_timestamp}_{state}.png`
- Imports ordenados: stdlib, third-party, local
- Docstrings só em funções públicas não-triviais; nunca docstring óbvia
- Nomes de funções em inglês; strings de prompt em PT-BR

## Regras
- Nunca rodar `agent.py` sem confirmar (consome GPU pesado)
- Nunca commitar `frames/`, `logs/`, `notes/`, `.env`, modelos
- Nunca tocar arquivos do Steam ou da pasta do jogo
- Nunca encher de docstring vazia ou comentário óbvio
- Antes de mudar prompts, ler `@docs/prompts.md`
- Antes de mudar mapeamento de gamepad ou regiões OCR, ler `@docs/coords.md`
- Status, fase, TODOs em `@docs/status.md` — atualizar ao concluir etapa
- Se VLM falhar parsing 3x seguidas, abortar turno e logar — não chutar ação
- OCR via pytesseract sempre antes de VLM para valores numéricos pequenos (HP, mana)
- Prompts ao VLM exigem transcrição literal, sem correção ortográfica

## Jogo
Vampire Crawlers, deckbuilder turn-based. UI em **português**, jogado com controle. Janela windowed 1280x720 auto-centralizada no monitor primário. Estados: combat, map, level_up, chest, chest_card_target, boss_chest, stage_complete, game_complete, shop, menu, game_over, title. Detalhes em `@docs/states.md`. Convenções de prompt em `@docs/prompts.md`. Mecânicas em `@jogo.md`.

## Pivot 2026-05-02
Input mudou de pyautogui+coords para vgamepad (DS4 virtual). Cartas/opções escolhidas por travessia (← →) com X. Em combate: scan sequencial (1 print por carta) + decisão UMA ação por vez. No mapa: pergunta de direção (frente/esq/dir/atrás) + micro-ação. Erros não acumulam — cada step recaptura.

## Performance
Latência alvo por turno: <15s. Se passar, reduzir resolução do screenshot enviado ao VLM (resize pra 768px no maior lado). Modelo carrega em ~30s na 1ª chamada — manter Ollama rodando entre sessões.

## Open source
Licença MIT. README com setup, demo gif, contribuição. Sem segredos no histórico git. Issues e PRs em inglês ou português, ambos OK.
