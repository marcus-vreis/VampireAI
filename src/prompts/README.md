# Prompts

Templates de prompt enviados ao VLM, organizados por estado do jogo.

## Padrão

- Um arquivo `.txt` por estado: `combat.txt`, `map.txt`, `menu.txt`, `level_up.txt`, `shop.txt`, `chest.txt`, `game_over.txt`, `title.txt`.
- Conteúdo em **PT-BR**, alinhado à UI do jogo (também em PT-BR).
- Exigir transcrição **literal** dos textos visíveis na tela, sem correção ortográfica.
- Pedir saída estruturada compatível com o schema pydantic correspondente em `src/schemas.py`.
- Placeholders entre chaves: `{hp}`, `{mana}`, `{notes}`, etc. — preenchidos no `src/perception.py`.
- Sem instruções de cadeia de raciocínio longas; manter prompts curtos pra latência <15s.

## Convenções

- Cabeçalho do prompt explica papel ("Você é um analista de tela do jogo Vampire Crawlers...").
- Lista de campos esperados na saída, com tipos.
- Exemplo mínimo de saída válida ao final, quando ajudar o modelo.
- Antes de editar prompts, ler `docs/prompts.md`.
