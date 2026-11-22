"""A rotulagem precisa capturar o JOGO, não o terminal — e gravar o que a pessoa
realmente respondeu, sem confundir "nenhuma" com "não sei"."""

from __future__ import annotations

from unittest import mock

import pytest

from src import label
from src.vision.screen import Verdict


def assinatura(verdict):
    return mock.Mock(verdict=verdict)


@pytest.fixture
def captura(tmp_path):
    shot = tmp_path / "x.png"
    shot.write_bytes(b"x")
    with (
        mock.patch.object(label, "grab", return_value=shot),
        mock.patch.object(label, "cv2"),
        mock.patch.object(label, "DATASET_DIR", tmp_path / "ds"),
        mock.patch.object(label, "LABELS_FILE", tmp_path / "ds" / "labels.jsonl"),
        mock.patch.object(label, "game_is_visible", return_value=True),
    ):
        yield shot


def test_recusa_gravar_quando_o_jogo_esta_coberto(captura):
    """Pra ler a tecla o terminal precisa do foco, então o jogo nunca está
    focado durante a rotulagem. Quem responde se dá pra capturar é o SISTEMA:
    a janela do jogo está no topo nos pixels que seriam capturados?

    Sem isso, um jogo atrás do terminal produziria um dataset inteiro de prints
    do terminal rotulados como combate."""
    with mock.patch.object(label, "game_is_visible", return_value=False):
        assert label.capture_labeled("combat") is None


def test_captura_tela_que_a_CV_NAO_reconhece(captura):
    """A regressão que motivou trocar o guarda (ADR-084).

    O guarda antigo perguntava "a CV reconhece esta tela?" — e recusava quando a
    resposta era não. Mas as telas que MAIS precisam ser rotuladas são
    exatamente essas: menu, loja, título, game over. O guarda recusava o
    material necessário pra consertar o próprio ponto cego.
    """
    with (
        mock.patch.object(label, "signature", return_value=assinatura(Verdict.NOT_GAME)),
        mock.patch.object(label, "_observed", return_value={"cv_verdict": "not_game"}),
    ):
        alvo = label.capture_labeled("shop")
        assert alvo is not None, "jogo visível: captura, mesmo sem a CV reconhecer a tela"
        assert label.registrar(alvo, "shop", {})["state"] == "shop"


def test_grava_quando_a_captura_e_o_jogo(captura):
    with (
        mock.patch.object(label, "signature", return_value=assinatura(Verdict.COMBAT)),
        mock.patch.object(label, "_observed", return_value={"cv_verdict": "combat"}),
    ):
        alvo = label.capture_labeled("combat")
        assert alvo is not None
        registro = label.registrar(alvo, "combat", {"hand_size": 5, "hand_size_known": True})
    assert registro["state"] == "combat"
    assert registro["hand_size"] == 5


def test_a_sessao_nao_comeca_com_o_jogo_coberto(captura, capsys):
    """Melhor recusar de saída que descobrir depois de 50 frames rotulados."""
    with mock.patch.object(label, "game_is_visible", return_value=False):
        assert label.session(ask_details=False) == 2


def test_sem_janela_localizada_cai_na_checagem_por_conteudo(captura):
    """`game_is_visible` devolve None quando o Win32 não acha a janela. Aí não há
    resposta do sistema e sobra a checagem fraca — com o ponto cego de volta,
    mas avisado. Melhor que capturar o terminal em silêncio."""
    with (
        mock.patch.object(label, "game_is_visible", return_value=None),
        mock.patch.object(label, "signature", return_value=assinatura(Verdict.NOT_GAME)),
    ):
        assert label.capture_labeled("combat") is None


def test_nenhuma_pergunta_aponta_pro_mesmo_campo_que_outra_do_mesmo_estado():
    """Duas perguntas do mesmo estado com o mesmo `cv` significam que uma delas
    está sendo medida contra a resposta errada. Foi o que aconteceu com
    `offered`, que apontava pra contagem da MÃO enquanto pergunta sobre o painel
    central de escolha — duas caixas diferentes, gabarito impossível."""
    for estado, perguntas in label._PERGUNTAS.items():
        alvos = [p.cv for p in perguntas]
        assert len(alvos) == len(set(alvos)), f"{estado} tem duas perguntas medindo {alvos}"


def test_o_observed_produz_todos_os_campos_que_as_perguntas_citam(tmp_path):
    """O contrato central: toda pergunta aponta pra um campo que a CV realmente
    devolve. Pergunta sem gabarito custa o tempo de quem responde e não mede
    nada — vira opinião solta no dataset."""
    frame = label.PROJECT_ROOT / "dataset" / "20260830T135106101_label_combat.png"
    if not frame.is_file():
        pytest.skip("frame de referência ausente")
    observado = set(label._observed(frame))
    citados = {p.cv for ps in label._PERGUNTAS.values() for p in ps}
    assert citados <= observado, f"perguntas sem gabarito: {citados - observado}"


def test_nenhuma_selecionada_e_resposta_nao_falta():
    """O caso que confundiu na primeira sessão: num combate com o cursor em
    "Finalizar turno" nenhuma carta está levantada. Isso é gabarito válido, e
    tratar como "não respondi" apagaria o caso mais comum de cursor nulo."""
    assert label._selecionada("n") == (None, True)
    assert label._selecionada("") == (None, False)


def test_a_pergunta_e_1_based_e_o_dado_e_0_based():
    """Contar a partir de 1 é natural pra quem olha a tela; o resto do código
    indexa a partir de 0. A conversão fica aqui, num lugar só."""
    assert label._selecionada("1") == (0, True)
    assert label._selecionada("4") == (3, True)


def test_hp_vira_lista_e_nao_tupla():
    """`read_hp` devolve tupla, o JSONL grava lista. Se o rótulo fosse tupla, o
    frame recarregado do disco compararia (61, 61) com [61, 61] e diria ERROU
    em cima de dois valores iguais."""
    assert label._par_hp("61/61") == ([61, 61], True)
    assert label._par_hp("61") == (None, False)


def test_direcao_e_sim_nao():
    assert label._direcao("n") == ("norte", True)
    assert label._direcao("o") == ("oeste", True)
    assert label._direcao("x") == (None, False)
    assert label._sim_nao("s") == (True, True)
    assert label._sim_nao("n") == (False, True)


def test_perguntas_de_combate_e_de_mapa_sao_diferentes():
    """Perguntar "quantas cartas" num frame de mapa gastaria o tempo de quem
    responde num campo que ali não significa nada."""
    combate = {p.campo for p in label._PERGUNTAS["combat"]}
    mapa = {p.campo for p in label._PERGUNTAS["map"]}
    assert "mana" in combate and "mana" not in mapa
    assert "facing" in mapa and "facing" not in combate


def test_rotulo_antigo_sem_flag_ainda_conta_como_resposta():
    """A primeira sessão gravou antes de existir o par `_known`. Ali, valor
    presente já significava resposta — o resumo não pode descartar esses frames."""
    antigo = {"hand_size": 6, "cursor": None}
    assert label._sabe(antigo, "hand_size") is True
    assert label._sabe(antigo, "cursor") is False
    assert label._sabe({"cursor": None, "cursor_known": True}, "cursor") is True


def test_a_acuracia_ignora_frames_sem_gabarito():
    """Um frame sem resposta não é acerto nem erro. Contar como erro puniria a
    CV por uma pergunta que ninguém respondeu."""
    pergunta = label._PERGUNTAS["combat"][1]
    records = [
        {"cursor": 2, "cursor_known": True, "cv_cursor": 2},
        {"cursor": None, "cursor_known": False, "cv_cursor": 9},
    ]
    assert "1/1" in label._acuracia(records, pergunta)


def test_a_ajuda_sai_da_mesma_tabela_das_perguntas():
    """Ajuda escrita à mão descreveria a versão anterior das perguntas na
    primeira vez que alguém mudasse a tabela, e ajuda errada é pior que nenhuma."""
    texto = label._como_responder()
    for p in label._PERGUNTAS["combat"]:
        assert p.campo in texto and p.onde[:20] in texto


def test_a_cobertura_diz_o_que_falta():
    """A pergunta "é pra fazer o jogo todo?" precisa ter resposta na tela."""
    assert "completa" in label._tabela_cobertura(dict.fromkeys(label._META, 99))
    assert "combat" in label._tabela_cobertura({"combat": 0})


def test_custo_da_carta_e_mana_disponivel_sao_perguntas_diferentes():
    """Duas coisas distintas que a mesma palavra "mana" nomeia: o que a carta
    CUSTA (algarismo no círculo, um por carta) e o que você TEM (orbe azul).
    Sem as duas não dá pra medir se `combat.validate` reprova jogada impossível."""
    combate = {p.campo: p for p in label._PERGUNTAS["combat"]}
    assert combate["costs"].cv == "cv_costs"
    assert combate["mana"].cv == "cv_mana"
    assert "CUSTA" in combate["costs"].onde and "TEM" in combate["mana"].onde


def test_lista_de_custos_aceita_virgula_e_espaco():
    assert label._lista_de_custos("1,1,1,2,0,0") == ([1, 1, 1, 2, 0, 0], True)
    assert label._lista_de_custos("1 1 2") == ([1, 1, 2], True)
    assert label._lista_de_custos("") == (None, False)


def test_a_cobertura_explica_o_que_variar():
    """ "São 12 fotos do combate?" — não: 12 SITUAÇÕES diferentes. Doze fotos do
    mesmo turno medem uma situação só, e o detector passaria sem ser testado."""
    texto = label._tabela_cobertura({"combat": 0})
    assert "VARIAÇÃO" in texto
    assert "mana alta e baixa" in texto


def test_o_gabarito_de_escolha_vem_do_painel_central_nao_da_mao():
    """`detect_card_slots` olha a caixa da MÃO, que só se sobrepõe em parte ao
    painel de escolha. Medir "quantas cartas na oferta" contra a contagem da mão
    culparia a CV por estar lendo a caixa errada."""
    escolha = {p.campo: p.cv for p in label._PERGUNTAS["level_up"]}
    assert escolha["offered"] == "cv_choice_cards"
    assert escolha["cursor"] == "cv_choice_cursor"


def test_a_pergunta_de_mao_mede_a_contagem_CORRIGIDA():
    """`visible_total` é piso: com carta levantada ele conta uma a menos, porque
    a levantada tapa o círculo da vizinha. Comparar a resposta humana contra o
    piso registraria erro da CV onde ela nem tentou responder."""
    combate = {p.campo: p.cv for p in label._PERGUNTAS["combat"]}
    assert combate["hand_size"] == "cv_hand_size"
