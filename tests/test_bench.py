"""Geração de cenários do bench. Não chama modelo."""

from __future__ import annotations

from src.bench import Tally, build_scenarios
from src.combat import validate


def test_cenarios_sao_reprodutiveis():
    """Mesma seed tem que dar os mesmos cenários — senão não dá pra comparar modelos."""
    a = build_scenarios(8, seed=7)
    b = build_scenarios(8, seed=7)
    assert [s.mana for s in a] == [s.mana for s in b]
    assert [[c.nome for c in s.hand] for s in a] == [[c.nome for c in s.hand] for s in b]


def test_todo_cenario_tem_jogada_possivel():
    """Cenário sem jogada legal não mede nada — o modelo acertaria por não ter escolha."""
    for scenario in build_scenarios(30, seed=3):
        assert scenario.playable
        assert scenario.by_rule is not None


def test_jogada_pela_regra_e_sempre_legal():
    for scenario in build_scenarios(30, seed=11):
        assert validate(scenario.by_rule, scenario.hand, scenario.mana) is None


def test_cenarios_variam_entre_seeds():
    a = build_scenarios(10, seed=1)
    b = build_scenarios(10, seed=2)
    assert [s.mana for s in a] != [s.mana for s in b]


def test_tally_calcula_percentual():
    t = Tally(total=25)
    t.legal = 23
    assert round(t.rate(t.legal)) == 92
    assert Tally(total=0).rate(0) == 0.0


def test_margem_encolhe_com_mais_cenarios():
    """Sem a margem o número é lido como mais preciso do que é: duas execuções
    de n=25 na mesma configuração deram 60% e 48% de aderência à regra."""
    pequena = Tally(total=25).margin(14)
    grande = Tally(total=100).margin(55)
    assert pequena > 15
    assert grande < pequena


def test_margem_zero_sem_cenarios():
    assert Tally(total=0).margin(0) == 0.0


def test_relatorio_mostra_a_margem(capsys):
    from src.bench import report

    t = Tally(total=50)
    t.parsed, t.legal, t.by_rule, t.latencies = 50, 46, 27, [1.6]
    report({"modelo-x": t})
    saida = capsys.readouterr().out
    assert "±" in saida
    assert "50 cenários" in saida
