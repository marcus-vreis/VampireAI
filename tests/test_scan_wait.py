"""Espera adaptativa pelo movimento do cursor durante a travessia da mão."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import cv2
import pytest

from src import perception
from src.vision.cards import CostCircle

_RAIZ = Path(__file__).resolve().parent.parent
# Gabarito versionado. `frames/` é gitignored E rotacionado durante uma run,
# então teste que dependa dele passa a pular em silêncio — pior que falhar.
FRAMES = _RAIZ / "dataset" / "referencia"
# Sequência real de scan: o cursor anda uma carta pra esquerda a cada passo.
SEQUENCIA = [
    "20260802T154032207_card_scan_1.png",
    "20260802T154035945_card_scan_2.png",
    "20260802T154039124_card_scan_3.png",
    "20260802T154042627_card_scan_4.png",
]


def caminho(name: str) -> Path:
    path = FRAMES / name
    if not path.is_file():
        pytest.skip(f"frame de referência ausente: {name}")
    return path


class FonteDeFrames:
    """Substitui `grab`: devolve os frames da sequência, um por chamada."""

    def __init__(self, names: list[str], repeticoes: int = 1) -> None:
        self.fila = [caminho(n) for n in names for _ in range(repeticoes)]
        self.chamadas = 0

    def __call__(self, state: str | None = None) -> Path:
        self.chamadas += 1
        return self.fila.pop(0) if self.fila else caminho(SEQUENCIA[-1])


def test_para_assim_que_o_cursor_anda():
    """Uma captura basta quando o jogo já respondeu — não espera tempo fixo."""
    fonte = FonteDeFrames([SEQUENCIA[1]])
    anterior = cv2.imread(str(caminho(SEQUENCIA[0])))
    x_anterior = perception.detect_card_slots(anterior).selected.x

    with mock.patch.object(perception, "grab", fonte), mock.patch.object(perception, "gamepad"):
        _frame, selecionada = perception._tap_and_wait(x_anterior)

    assert fonte.chamadas == 1
    assert selecionada is not None
    assert abs(selecionada.x - x_anterior) >= perception._SAME_CARD_PX


def test_insiste_enquanto_o_cursor_nao_saiu_do_lugar():
    """O sleep fixo lia um frame ainda em animação; aqui a captura se repete."""
    fonte = FonteDeFrames([SEQUENCIA[0], SEQUENCIA[0], SEQUENCIA[1]])
    anterior = cv2.imread(str(caminho(SEQUENCIA[0])))
    x_anterior = perception.detect_card_slots(anterior).selected.x

    with mock.patch.object(perception, "grab", fonte), mock.patch.object(perception, "gamepad"):
        _frame, selecionada = perception._tap_and_wait(x_anterior)

    assert fonte.chamadas == 3
    assert abs(selecionada.x - x_anterior) >= perception._SAME_CARD_PX


def test_desiste_no_teto_quando_o_cursor_nunca_anda():
    """Na ponta do leque o cursor não se move; a travessia tem que terminar."""
    fonte = FonteDeFrames([SEQUENCIA[0]], repeticoes=200)
    anterior = cv2.imread(str(caminho(SEQUENCIA[0])))
    x_anterior = perception.detect_card_slots(anterior).selected.x

    with (
        mock.patch.object(perception, "grab", fonte),
        mock.patch.object(perception, "gamepad"),
        mock.patch.object(perception, "_CURSOR_MOVE_TIMEOUT_S", 0.15),
    ):
        _frame, selecionada = perception._tap_and_wait(x_anterior)

    assert selecionada is not None
    assert abs(selecionada.x - x_anterior) < perception._SAME_CARD_PX


def test_travessia_completa_percorre_a_mao_uma_vez():
    """A travessia para sozinha ao voltar a uma carta já vista — sem saber o total.

    O primeiro frame vem duplicado porque `_wait_for_hand` consome um PAR de
    capturas antes de começar: ele espera a contagem repetir, que é como distingue
    a mão assentada da mesa em animação (ADR-090). A travessia em si continua
    gastando um frame por carta.
    """
    fonte = FonteDeFrames([SEQUENCIA[0]] + SEQUENCIA + [SEQUENCIA[-1]])
    lido = CostCircle(x=0, y=0, w=20, h=20)
    with (
        mock.patch.object(perception, "grab", fonte),
        mock.patch.object(perception, "gamepad"),
        mock.patch.object(perception, "read_card") as ler,
        mock.patch.object(perception, "read_mana_hybrid", return_value=3),
        mock.patch.object(perception, "read_hp_hybrid", return_value=(61, 61)),
    ):
        ler.side_effect = lambda *a, **k: perception.CardScanFrame(
            nome=f"carta{ler.call_count}", mana=1, descricao=None, tipo="ataque"
        )
        scan = perception.scan_combat_hand()

    assert len(scan.cards) == len(SEQUENCIA)
    assert scan.mana == 3
    assert scan.hp == (61, 61)
    assert lido is not None


# --- entrada na mao --------------------------------------------------------
# No comeco do turno NENHUMA carta esta levantada: o cursor fica fora do leque,
# sobre "Finalizar turno". A travessia assumia que ja havia uma selecionada e
# desistia no passo 0, encerrando o turno sem jogar nada. Ver ADR-088.


def test_a_maioria_dos_frames_de_combate_nao_tem_carta_levantada():
    """A medicao que motivou o conserto: o estado que o codigo assumia como
    INICIAL era na verdade um estado intermediario que ele mesmo produzia."""
    import json

    from src.vision.cards import detect_card_slots

    rotulos = _RAIZ / "dataset" / "labels.jsonl"
    if not rotulos.is_file():
        pytest.skip("dataset sem rotulos")
    combates = [
        json.loads(linha)["file"]
        for linha in rotulos.read_text(encoding="utf-8").splitlines()
        if linha.strip() and json.loads(linha)["state"] == "combat"
    ]
    if len(combates) < 5:
        pytest.skip("poucos frames de combate")
    sem = sum(
        1
        for n in combates
        if detect_card_slots(cv2.imread(str(_RAIZ / "dataset" / n))).selected_idx is None
    )
    assert sem > len(combates) / 2, (
        "se a maioria passasse a ter carta levantada, a premissa do _enter_hand mudou"
    )


def test_entra_na_mao_quando_nenhuma_carta_esta_levantada():
    """O conserto: em vez de desistir, poe o cursor no leque e so entao percorre."""
    alvo = CostCircle(x=400, y=500, w=30, h=30)
    respostas = [(None, None), (None, alvo)]
    with mock.patch.object(perception, "_tap_and_wait", side_effect=respostas) as tap:
        _, selected = perception._enter_hand(None)
    assert selected is alvo
    assert tap.call_count == 2, "insiste ate o cursor aparecer"


def test_desiste_de_entrar_na_mao_depois_de_alguns_toques():
    """Mao vazia existe: sem teto, a travessia ficaria tocando pra sempre."""
    with mock.patch.object(perception, "_tap_and_wait", return_value=(None, None)) as tap:
        _, selected = perception._enter_hand(None)
    assert selected is None
    assert tap.call_count == perception._ENTER_HAND_TAPS


def test_a_sentinela_faz_qualquer_selecao_contar_como_movimento():
    """`_tap_and_wait` espera o cursor SAIR de uma posicao. Sem cursor nenhum,
    nao ha posicao de onde sair — a sentinela resolve isso sem caso especial."""
    assert perception._NO_CURSOR < -10_000


# --- animacao de distribuicao ----------------------------------------------
# O jogo distribui a mao com animacao: as cartas voam de baixo pra cima. Capturar
# durante isso mostra a mesa vazia. Medido nos frames da run 3: 20 de 42 inicios
# de travessia (48%) pegaram a mesa em animacao. Ver ADR-090.


def _slots(n, selected=None):
    circles = [CostCircle(x=300 + 100 * i, y=500, w=22, h=22) for i in range(n)]
    return perception.CardSlots(circles=circles, selected_idx=selected)


def test_espera_a_contagem_REPETIR_nao_so_aparecer():
    """Um frame no meio da animacao ja mostra algumas cartas. "Achou carta" nao
    basta como sinal de que a mao assentou -- a contagem tem que estabilizar."""
    leituras = [_slots(0), _slots(2), _slots(5), _slots(6), _slots(6)]
    with (
        mock.patch.object(perception, "grab", return_value=Path("x.png")),
        mock.patch.object(perception.cv2, "imread", return_value=None),
        mock.patch.object(perception, "detect_card_slots", side_effect=leituras),
    ):
        _, slots = perception._wait_for_hand()
    assert slots.visible_total == 6, "parou na primeira contagem que se repetiu"


def test_desiste_quando_a_mao_esta_mesmo_vazia():
    """Mao vazia existe -- todas as cartas jogadas. Sem teto o agente travaria
    esperando cartas que nao vem."""
    with (
        mock.patch.object(perception, "grab", return_value=Path("x.png")),
        mock.patch.object(perception.cv2, "imread", return_value=None),
        mock.patch.object(perception, "detect_card_slots", return_value=_slots(0)),
        mock.patch.object(perception, "_HAND_DEAL_TIMEOUT_S", 0.05),
    ):
        _, slots = perception._wait_for_hand()
    assert slots.visible_total == 0


def test_zero_nunca_conta_como_contagem_estavel():
    """Duas leituras de zero seguidas sao a mesa em animacao, nao mao vazia --
    e exatamente o caso que fazia o agente encerrar o turno com a mao cheia.
    So o teto decide que a mao esta vazia, nunca a repeticao do zero."""
    with (
        mock.patch.object(perception, "grab", return_value=Path("x.png")),
        mock.patch.object(perception.cv2, "imread", return_value=None),
        mock.patch.object(
            perception, "detect_card_slots", side_effect=[_slots(0)] * 3 + [_slots(6), _slots(6)]
        ),
        mock.patch.object(perception, "_HAND_DEAL_TIMEOUT_S", 5.0),
    ):
        _, slots = perception._wait_for_hand()
    assert slots.visible_total == 6, "atravessou tres zeros ate a mao assentar"
