# Coordenadas e mapeamento de input

## Janela do jogo
Windowed 1280x720, centralizada no monitor primário. `src/config.py` detecta o monitor primário em runtime via `mss` e calcula `(x, y) = ((mon_w - 1280) / 2, (mon_h - 720) / 2)`. Override via `GAME_WINDOW_X` / `GAME_WINDOW_Y` no `.env`.

A janela importa apenas para **captura** (`mss.grab(region)`). Não há mais clique de mouse.

## Input — gamepad virtual

**Mudança fundamental (2026-05-02):** abandonado mouse + coords de UI. Input agora é DualShock 4 virtual via [`vgamepad`](https://github.com/yannbouteiller/vgamepad) + driver [ViGEm Bus](https://vigem.org/). O jogo aceita controle nativamente.

### Por que gamepad
- Cartas/opções selecionadas por **destaque visual** (a "maior/mais à frente"), não por posição XY → não precisa medir nenhum slot.
- Navegação por travessia: ← e → no D-pad para mover o cursor.
- `vgamepad` simula no nível de driver — o jogo recebe igual a um controle físico.

### Mapeamento de botões (jogo.md)
| Ação no jogo | Botão no controle | Função em código |
|---|---|---|
| Mover pra cima/baixo/esquerda/direita | D-pad ↑ ↓ ← → | `walk_forward`, `walk_back`, `walk_left`, `walk_right` |
| Girar pra direita | R2 | `turn_right` |
| Girar pra esquerda | L2 | `turn_left` |
| Confirmar / jogar carta / escolher | X (cross) | `confirm` |
| Sacar dinheiro / cancelar | Quadrado | `cancel` |

### Timing
Em `config.py::GamepadConfig`:
- `press_hold_s` = 0.08 (tempo entre press/release de UM botão)
- `between_actions_s` = 0.25 (entre ações consecutivas)
- `post_dpad_settle_s` = 0.4 (depois de navegar, espera UI estabilizar antes de X)
- `boot_delay_s` = 0.5 (espera após inicializar o gamepad virtual)

`jogo.md` recomenda ~1s entre ações compostas. `between_actions_s` cobre isso quando somado a `post_dpad_settle_s`.

## Regiões de captura (OCR auxiliar)
Mantemos OCR só para HP/mana globais do HUD (posição fixa no frame). Cartas/opções não usam OCR — VLM lê do print da carta isolada.

| Região | (x, y, w, h) | Conteúdo |
|---|---|---|
| `hp_player` | (185, 505, 45, 22) | HP atual |
| `hp_max` | (185, 530, 45, 22) | HP máximo |
| `mana` | (1060, 525, 30, 30) | Mana atual |
| `mana_max` | TBD | Mana máxima (pode não ser visível) |

Coords TBD ficam como `(-1, -1, -1, -1)` e `is_region_set` retorna False — OCR pula essa região.

## Como atualizar
1. `python -m src.capture --once` salva frame em `frames/`.
2. Abrir em editor que mostra coords (Paint.NET, GIMP).
3. Atualizar `OCR_REGIONS` em `src/config.py` e a tabela acima.
