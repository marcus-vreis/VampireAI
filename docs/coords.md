# Regiões de UI

Input é gamepad — **não existem coordenadas de clique**. As regiões abaixo são só
recortes de imagem para a visão computacional.

## Referencial

Todas as caixas são medidas contra o **client area** da janela do jogo, 1280x720.
`src/window.py` localiza esse retângulo via Win32 a cada captura
(`EnumWindows` → `GetClientRect` → `ClientToScreen`).

Isso importa: antes a captura era um retângulo calculado (monitor centralizado
menos 1280x720). Frames de sessões diferentes saíam com geometrias distintas
conforme a janela tivesse barra de título ou fosse movida, e todo recorte saía do
lugar junto. Ver ADR-023.

Se a janela não for encontrada, `find_game_window` cai no retângulo de `config.py`
e avisa no log. Conferir com:

```bash
python -m src.window
```

## Caixas em uso

Definidas em `src/vision/regions.py` (fração do client area) e replicadas como
constantes de módulo onde são usadas em caminho quente.

| Nome | Pixels (1280x720) | Para quê |
|---|---|---|
| `hand` | `(235, 380)-(1035, 720)` | leque de cartas: círculos de custo e cursor |
| `minimap` | `(705, 512)-(1010, 720)` | pergaminho, seta do jogador, fronteira |
| `mana_orb` | `(1015, 455)-(1145, 585)` | orbe de mana |
| `hp_heart` | `(140, 450)-(275, 580)` | coração de HP |
| `viewport` | `(258, 28)-(1022, 512)` | visão 3D — inimigos e corredor |
| `choice_area` | `(280, 140)-(1010, 560)` | painel central das telas de escolha |

### Nota sobre `hand`

A caixa anterior era `(380, 460, 480, 260)`, ou seja `(380,460)-(860,720)`. Num
frame real de 6 cartas ela cortava a carta da ponta esquerda inteira e fatiava ao
meio a selecionada. Como o scan de cartas usava o mesmo recorte, o VLM lia um
pedaço da carta. Foi a causa raiz da contagem errada e da piora na leitura.

A caixa atual tem folga deliberada nas duas pontas.

## Assinaturas de cor

Amostradas em frames reais. Alterar sem remedir quebra a detecção.

| Elemento | Faixa HSV (OpenCV, H 0-179) | Onde |
|---|---|---|
| Círculo de custo, normal | H 100-125, S 150-255, V 60-230 | `vision/cards.py` |
| Círculo de custo, destacado | H 138-176, S 90-255, V 90-255 | idem — a carta selecionada **pulsa** entre azul e magenta |
| Seta do jogador | H 100-125, S 150-255, V 80-255 | `vision/minimap.py` |
| Pergaminho do minimapa | cinza 150-235 | `vision/screen.py` |
| Piso conhecido no minimapa | cinza > 185 | `vision/minimap.py` |
| Névoa (não revelado) | cinza 140-185 | `vision/minimap.py` |
| Ícones (caveira, chefe, bônus, ?) | cinza 128-146 | `vision/minimap.py`, `vision/icons.py` |
| Painel de diálogo | H 100-135, S 20-110, V 60-190 | `vision/screen.py` |
| Texto claro do HUD | S < 90, V > 185 | `vision/hud.py` |

## Escalas medidas

- Círculo de custo: ~22px de lado; a carta selecionada, ~34px. A razão é o que
  identifica o cursor (`_SELECTED_SIDE_RATIO = 1.25`).
- Carta: ~7.4x o lado do círculo em largura, ~9.2x em altura.
- Seta do jogador: 16px numa fase, 19px noutra — **o minimapa muda de zoom entre
  fases**. Por isso a navegação não assume tamanho de célula.

## Níveis do minimapa

Medidos num frame real. A separação é limpa, com um vale vazio entre névoa e piso:

| Elemento | Cinza |
|---|---|
| Vazio fora do mapa | 0-49 |
| Ícones | 136 (valor único) |
| Névoa | 150-179 |
| *(vale — quase nenhum pixel)* | 180-189 |
| Piso conhecido | 194-206 |

Os ícones ficam ABAIXO do limiar de piso. Eles precisam ser somados à máscara
andável, senão viram buracos e o BFS não alcança o inimigo.

## Templates de ícone

`src/vision/templates/` guarda sprites recortados de frame real:

| Arquivo | Tamanho | O quê |
|---|---|---|
| `skull.png` | 13x13 | inimigo comum |
| `boss.png` | 21x17 | chefe (caveira com chifres) |
| `question.png` | 9x15 | ponto de interrogação |

A busca varre escala **absoluta** de 0.80 a 1.60 em passo de 0.05. Não derive a
escala do tamanho da seta do jogador: num frame ela mede 19px (razão 1.19)
enquanto os ícones pedem 1.30. Sendo pixel art, 0.05 de erro derruba a correlação
de 0.85 pra 0.60.

## Algarismos do HUD

Medidos no coração do frame de referência, que mostra 61/61:

| Componente | Tamanho | Densidade |
|---|---|---|
| Algarismo | 8-13 x 17-18 | 0.52-0.62 |
| Contorno do coração | 44x77 | 0.09 |

A **densidade** (área do componente sobre área da caixa) é o que separa os dois:
o contorno passa nos filtros de tamanho e proporção, mas é uma curva fina numa
caixa grande. Corte em 0.30, com margem de 3x pros dois lados.

O agrupamento em linhas usa a **altura mediana do algarismo** como referência. Não
use o espalhamento total: no coração as duas linhas ficam a 22px e o espalhamento
é 47, o que colapsa tudo numa linha só.

## Calibrar

```bash
python -m src.perception --cards frames/algum.png
```

```bash
python -m src.states --frame frames/algum.png --cv-only
```

Para ver tudo de uma vez, anotado sobre o frame — cartas, cursor, caixa do
minimapa, piso andável, posição e direção do jogador, ícones:

```bash
python -m src.vision.debug frames/algum.png
```

Use isto **antes** de mexer em qualquer limiar. Durante o desenvolvimento deste
módulo, olhar a máscara resolveu em uma tentativa o que várias rodadas de ajuste
de número às cegas não tinham resolvido.
