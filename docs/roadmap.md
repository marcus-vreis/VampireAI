# Roadmap

Projeto incremental em 6 fases. Cada fase fecha com entregável visível.

## Fase 0 — Captura (concluída no setup)
Screenshot da janela do jogo, salva em `frames/`. Pipeline VLM testado isolado.
**Entregável:** screenshot funcional + `python -m src.llm --ping` retorna resposta.

## Fase 1 — Input automation
Mouse e teclado simulados via pyautogui. Sequência fixa hardcoded para validar que botões chegam ao jogo.
**Entregável:** GIF curto de script abrindo o jogo e clicando em "iniciar run".

## Fase 2 — Percepção estruturada
Detector de estado + extração de cartas/HP/mana via VLM (com prompt cirúrgico) + OCR (pytesseract) para números. Saída em pydantic schemas.
**Entregável:** `python -m src.perception --frame X.png` retorna JSON estruturado completo do estado.

## Fase 3 — Combate fechado
Loop completo só em combate. VLM decide ordem de cartas via tool use, parser converte para coords, executor clica. Sem memória ainda.
**Entregável:** vídeo de 1-2 min do agente vencendo um combate. Post de LinkedIn.

## Fase 4 — Run completa
Múltiplos estados (mapa, level up, loja, baú). Memória persistente em `notes.md`. Sumarização accordion a cada N turnos.
**Entregável:** timelapse de 3-5 min de uma run completa. Thread técnica.

## Fase 5 — Ângulo de pesquisa
Escolher uma direção:
- A: comparar modelos open-weight pequenos (Qwen2.5-VL vs InternVL vs Llama 3.2 Vision)
- B: ablation de componentes do harness (sem memória, sem OCR, sem detector de estado)
- C: estresse de generalização (dificuldades altas, runs longas)

**Entregável:** post técnico longo, repo limpo com README sério, gráficos.

## Fase 6 — Publicação (opcional)
Workshop paper, com baseline rigoroso, múltiplas seeds, intervalos de confiança.
**Entregável:** paper 4-8 páginas em workshop NeurIPS/ICLR/BRACIS/ENIAC.

## Princípios
- Não engenharia: cada fase resolve um problema concreto da anterior
- Sem otimização prematura: medir antes de mexer
- Cada fase tem entregável compartilhável; se parar aqui, ainda tem valor
