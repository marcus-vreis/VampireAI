# Roadmap

Projeto incremental. Cada fase fecha com entregável visível.

Status detalhado e TODOs em [`status.md`](status.md). Decisões e o porquê delas em
[`decisions.md`](decisions.md).

## Fase 0 — Captura ✅
Screenshot da janela do jogo. Pipeline de modelo testado isolado.
**Entregável:** `python -m src.capture --once` e `python -m src.llm --ping`.

Evoluiu: a captura passou de retângulo calculado para o **client area real** da
janela via Win32 (ADR-023). O retângulo calculado deixava o frame deslocado
conforme a janela tivesse barra de título ou fosse movida, e todo recorte de UI
saía do lugar junto.

## Fase 1 — Input ✅
Gamepad virtual: `vgamepad` → driver ViGEm Bus → DualShock 4 emulado (ADR-014).

Substituiu mouse + coordenadas. Cartas e opções são escolhidas por destaque
visual ("a maior") + travessia (← →), o que eliminou toda a calibração de UI.
**Entregável:** `python -m src.gamepad --test` movimenta o jogo.

## Fase 2 — Percepção ✅
**Reescrita na ADR-022.** A percepção geométrica saiu do modelo e foi pra CV:
estado da tela, contagem de cartas, cursor, mana e navegação vêm de
`src/vision/`, em ~19ms e sem alucinar. O modelo ficou com semântica.

Medir foi o que forçou a mudança: de 39 frames que eram o mapa, o VLM rotulava ao
menos 9 como outra coisa.
**Entregável:** `python -m src.vision.debug frames/x.png` anota o frame com tudo
que a CV enxerga.

## Fase 3 — Combate ✅
Travessia da mão (não precisa saber o total antes, ADR-024), cache de carta por
hash perceptual (ADR-025), e jogada do modelo **validada** contra mana e índices
antes de executar (ADR-026).
**Entregável:** vídeo do agente vencendo um combate. *Pendente: rodar contra o
jogo aberto — todo o sensor novo foi validado só sobre frames salvos.*

## Fase 4 — Run completa 🔜 (fase atual)
Meta: **zerar a fase 1 do jogo.**

Falta:
- Sessão de rotulagem (`python -m src.label`) pra virar suíte de regressão.
- Rodar ponta a ponta contra o jogo.
- Templates de baú e obstáculo no minimapa.
- Features de aresta no mapa: bônus encostados em parede são desenhados
  deslocados pra borda da célula, não no centro.

**Entregável:** timelapse de uma run completa.

## Fase 5 — Ângulo de pesquisa
A infraestrutura já existe, o que muda a pergunta de "como medir?" pra "o que
medir?".

- **A — comparar modelos.** `python -m src.bench --models a,b` já roda, com
  gabarito derivado (a legalidade de uma jogada é regra que o código conhece,
  então não precisa de rótulo humano). Baseline registrada na ADR-032.
- **B — ablation do harness.** A separação CV/modelo é uma costura limpa: dá pra
  medir o custo de devolver cada peça ao modelo. Quanto piora sem o detector de
  CV? Sem o cache de carta? Sem o validador de jogada?
- **C — generalização.** Dificuldades altas, fases avançadas, runs longas.

O ângulo mais forte hoje é o **B**: o projeto tem o antes e o depois medidos no
mesmo jogo, com o mesmo modelo. É evidência direta de que o andaime importa mais
que o tamanho do modelo — que é justamente a lição que motivou o projeto.

**Entregável:** post técnico longo, gráficos, repo limpo.

## Fase 6 — Publicação (opcional)
Workshop paper com baseline rigoroso, múltiplas seeds, intervalos de confiança.
**Entregável:** paper de 4-8 páginas.

## Princípios
- Não engenharia: cada fase resolve um problema concreto da anterior
- **Medir antes de mexer.** Todo limiar deste projeto tem número que o justifica
- **Olhar a máscara antes de afinar limiar.** Ajustar número às cegas custa mais
  que renderizar o overlay e ver o que está sendo pego
- Cada fase tem entregável compartilhável; se parar aqui, ainda tem valor
