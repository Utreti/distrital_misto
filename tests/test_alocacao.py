"""Testes do motor proporcional: valores da seção 4.1 e propriedades genéricas."""

import logging
import random
from fractions import Fraction

import pandas as pd
import pytest

import alocacao
import carga
import pipeline


@pytest.fixture(scope="module")
def detalhe(votos_ag, params):
    return alocacao.detalhar_quociente_atual(votos_ag, params["n_cadeiras"], barreira=True)


def _distribuidas(serie):
    return serie[serie > 0].to_dict()


# Valores conferidos à mão no Acre 2022 (seção 4.1).


def test_quociente_eleitoral(detalhe):
    assert detalhe.qe == 54_282


def test_quociente_partidario(detalhe):
    assert _distribuidas(detalhe.quociente_partidario) == {
        "PP": 1,
        "UNIÃO": 1,
        "REPUBLICANOS": 1,
    }


def test_agremiacoes_acima_da_barreira(detalhe, votos_ag):
    assert votos_ag[list(detalhe.elegiveis)].to_dict() == {
        "PP": 93_602,
        "REPUBLICANOS": 59_017,
        "UNIÃO": 84_939,
    }


def test_ordem_das_sobras_com_barreira(detalhe):
    assert detalhe.ordem_sobras == ("PP", "UNIÃO", "PP", "REPUBLICANOS", "UNIÃO")


def test_alocacao_final_com_barreira(votos_ag):
    cadeiras = alocacao.alocar_proporcional(votos_ag, 8, "quociente_atual", barreira=True)
    assert _distribuidas(cadeiras) == {"PP": 3, "REPUBLICANOS": 2, "UNIÃO": 3}


def test_igual_ao_resultado_oficial(dados, votos_ag):
    cadeiras = alocacao.alocar_proporcional(votos_ag, 8, "quociente_atual", barreira=True)
    assert _distribuidas(cadeiras) == carga.contar_eleitos_oficiais(dados["candidatos"]).to_dict()


def test_coligacao_distribui_vagas_aos_seus_mais_votados():
    votos = pd.Series({"A": 60, "B": 40, "C": 100})
    coligacoes = pd.Series({"A": 1, "B": 1, "C": 2})
    candidatos = pd.DataFrame(
        {"agremiacao": ["A", "B", "B", "C", "C"], "votos": [10, 50, 30, 90, 10]}
    )
    cadeiras = alocacao.alocar_com_coligacoes(votos, candidatos, coligacoes, 4)
    assert cadeiras.to_dict() == {"A": 0, "B": 2, "C": 2}


def test_coligacao_sem_candidato_com_10_por_cento_perde_a_vaga():
    votos = pd.Series({"A": 300, "B": 700})
    coligacoes = pd.Series({"A": 1, "B": 2})
    candidatos = pd.DataFrame({"agremiacao": ["A", "B", "B"], "votos": [10, 400, 300]})
    assert alocacao.alocar_com_coligacoes(votos, candidatos, coligacoes, 2).to_dict() == {"A": 0, "B": 2}


def test_validacao_do_pipeline_aprova_2022(dados):
    pipeline.validar_2022(dados["votos_ag"], dados["candidatos"], 8)


def test_validacao_do_pipeline_rejeita_divergencia(dados):
    candidatos = dados["candidatos"].copy()
    candidatos.loc[candidatos["DS_SIT_TOT_TURNO"] == "ELEITO POR QP", "agremiacao"] = "MDB"
    with pytest.raises(ValueError, match="difere"):
        pipeline.validar_2022(dados["votos_ag"], candidatos, 8)


# Regra compensatória (2.3) e número de distritos (2.4).


def test_alocacao_sem_barreira_no_acre(votos_ag):
    """Conferido à mão: sobras por maior média entre todas as agremiações."""
    cadeiras = alocacao.alocar_proporcional(votos_ag, 8, "quociente_atual", barreira=False)
    assert _distribuidas(cadeiras) == {
        "MDB": 1,
        "PP": 2,
        "PSDB/CIDADANIA": 1,
        "PT/PC do B/PV": 1,
        "REPUBLICANOS": 1,
        "UNIÃO": 2,
    }


@pytest.mark.parametrize(
    ("proporcao", "esperado"), [("1/2", 4), ("2/3", 6)]
)
def test_numero_de_distritos_do_acre(proporcao, esperado):
    assert alocacao.calcular_n_distritos(8, Fraction(proporcao)) == esperado


def test_numero_de_distritos_arredonda_para_cima():
    assert alocacao.calcular_n_distritos(9, Fraction("1/2")) == 5
    assert alocacao.calcular_n_distritos(10, Fraction("2/3")) == 7


def test_pec_compensa_sobre_todas_as_cadeiras():
    cadeiras_e = pd.Series({"A": 5, "B": 3})
    distritos_d = pd.Series({"A": 2, "B": 2})
    df = alocacao.combinar_pec(cadeiras_e, distritos_d)
    assert df["lista"].to_dict() == {"A": 3, "B": 1}
    assert df["total"].to_dict() == {"A": 5, "B": 3}
    assert df["excedente"].sum() == 0


def test_pec_gera_excedente_sem_limite():
    cadeiras_e = pd.Series({"A": 2, "B": 2})
    distritos_d = pd.Series({"A": 4})
    df = alocacao.combinar_pec(cadeiras_e, distritos_d)
    assert df.loc["A"].to_dict() == {
        "cadeiras_E": 2,
        "distritos_D": 4,
        "lista": 0,
        "excedente": 2,
        "total": 4,
    }
    assert df["total"].sum() == 6


def test_pec_preserva_distrito_de_agremiacao_com_e_zero():
    df = alocacao.combinar_pec(pd.Series({"A": 8, "B": 0}), pd.Series({"B": 1}))
    assert df.loc["B", "total"] == 1
    assert df.loc["B", "excedente"] == 1


# Regra real de lista aberta, usada só na validação.


def test_lista_aberta_reproduz_o_oficial_no_acre(dados, votos_ag):
    cadeiras = alocacao.alocar_lista_aberta(votos_ag, dados["votos_candidato"], 8)
    assert _distribuidas(cadeiras) == carga.contar_eleitos_oficiais(dados["candidatos"]).to_dict()


def test_lista_aberta_sem_candidato_com_20_porcento_perde_a_sobra():
    votos = pd.Series({"A": 700, "B": 300})
    # QE = 250: A tem 2 quocientes e um único candidato com 20% do QE além dos eleitos.
    candidatos = pd.DataFrame(
        {
            "agremiacao": ["A", "A", "B", "B"],
            "votos": [400, 300, 200, 100],
        }
    )
    cadeiras = alocacao.alocar_lista_aberta(votos, candidatos, 4)
    assert cadeiras.to_dict() == {"A": 2, "B": 2}


def test_lista_aberta_sem_candidato_com_10_porcento_perde_quociente():
    votos = pd.Series({"A": 1_000, "B": 1_000, "C": 500})
    # QE = 500: A tem 2 quocientes, mas só um candidato atinge 10% do QE; a sobra vai para C,
    # a única com candidato não eleito acima de 20% do QE.
    candidatos = pd.DataFrame(
        {
            "agremiacao": ["A", "A", "B", "B", "C", "C"],
            "votos": [960, 40, 500, 500, 400, 100],
        }
    )
    cadeiras = alocacao.alocar_lista_aberta(votos, candidatos, 5)
    assert cadeiras.to_dict() == {"A": 1, "B": 2, "C": 2}


# Propriedades genéricas do motor.


@pytest.mark.parametrize("barreira", [True, False])
@pytest.mark.parametrize("semente", range(20))
def test_cadeiras_sempre_somam_n(semente, barreira):
    gerador = random.Random(semente)
    votos = pd.Series(
        {f"AG{i}": gerador.randint(0, 100_000) for i in range(gerador.randint(2, 15))}
    )
    votos["AG0"] += 100_000
    n_cadeiras = gerador.randint(1, 70)
    cadeiras = alocacao.alocar_proporcional(votos, n_cadeiras, "quociente_atual", barreira)
    assert cadeiras.sum() == n_cadeiras


def test_sem_barreira_partido_sem_votos_nao_recebe_cadeira():
    votos = pd.Series({"A": 1_000, "B": 500, "C": 0})
    assert alocacao.alocar_proporcional(votos, 10, "quociente_atual", barreira=False)["C"] == 0


def test_sem_barreira_partido_pequeno_disputa_sobras():
    # QE = 500; C fica abaixo de 80% do QE, mas tem a maior média na última sobra.
    votos = pd.Series({"A": 1_100, "B": 520, "C": 380})
    com = alocacao.alocar_proporcional(votos, 4, "quociente_atual", barreira=True)
    sem = alocacao.alocar_proporcional(votos, 4, "quociente_atual", barreira=False)
    assert com.to_dict() == {"A": 3, "B": 1, "C": 0}
    assert sem.to_dict() == {"A": 2, "B": 1, "C": 1}


@pytest.mark.parametrize("barreira", [True, False])
def test_resultado_independe_da_ordem_de_entrada(votos_ag, barreira):
    embaralhado = votos_ag.sample(frac=1, random_state=7)
    pd.testing.assert_series_equal(
        alocacao.alocar_proporcional(embaralhado, 8, "quociente_atual", barreira),
        alocacao.alocar_proporcional(votos_ag, 8, "quociente_atual", barreira),
    )


def test_sem_elegiveis_sobras_vao_para_todas():
    votos = pd.Series({"A": 100, "B": 100, "C": 100, "D": 100, "E": 100})
    detalhe = alocacao.detalhar_quociente_atual(votos, 2, barreira=True)
    assert detalhe.elegiveis == ()
    assert detalhe.cadeiras.sum() == 2


def test_empate_de_medias_vai_para_mais_votos():
    votos = pd.Series({"A": 200, "B": 400})
    cadeiras = alocacao.alocar_proporcional(votos, 2, "quociente_atual", barreira=False)
    assert cadeiras.to_dict() == {"A": 0, "B": 2}


def test_empate_persistente_usa_nome_e_registra_aviso(caplog):
    votos = pd.Series({"B": 100, "A": 100})
    with caplog.at_level(logging.WARNING):
        cadeiras = alocacao.alocar_proporcional(votos, 1, "quociente_atual", barreira=False)
    assert cadeiras.to_dict() == {"A": 1, "B": 0}
    assert "Empate persistente" in caplog.text


@pytest.mark.parametrize(
    ("total", "n_cadeiras", "esperado"),
    [(434_253, 8, 54_282), (12, 8, 1), (13, 8, 2), (20, 8, 2), (10, 4, 2), (11, 4, 3)],
)
def test_arredondamento_do_qe(total, n_cadeiras, esperado):
    assert alocacao.calcular_qe(total, n_cadeiras) == esperado


def test_votos_insuficientes_geram_erro():
    with pytest.raises(ValueError, match="insuficientes"):
        alocacao.alocar_proporcional(pd.Series({"A": 1}), 8, "quociente_atual", False)


@pytest.mark.parametrize(
    ("metodo", "esperado"),
    [("dhondt", {"A": 4, "B": 3, "C": 1, "D": 0}), ("sainte_lague", {"A": 3, "B": 3, "C": 1, "D": 1})],
)
def test_metodos_de_divisores_em_exemplo_classico(metodo, esperado):
    votos = pd.Series({"A": 100_000, "B": 80_000, "C": 30_000, "D": 20_000})
    assert alocacao.alocar_proporcional(votos, 8, metodo, False).to_dict() == esperado


@pytest.mark.parametrize("metodo", ["dhondt", "sainte_lague"])
def test_barreira_exclui_quem_nao_tem_80_por_cento_do_qe(metodo):
    # QE = 100; C tem 79 votos e D tem exatamente 80.
    votos = pd.Series({"A": 441, "B": 400, "C": 79, "D": 80})
    com_barreira = alocacao.alocar_proporcional(votos, 10, metodo, True)
    assert com_barreira["C"] == 0
    assert com_barreira.sum() == 10
    assert alocacao.alocar_proporcional(votos, 10, metodo, False)["C"] >= com_barreira["C"]


def test_sem_agremiacao_elegivel_a_barreira_nao_se_aplica_nos_divisores():
    votos = pd.Series({"A": 3, "B": 3, "C": 3, "D": 3, "E": 2})
    assert alocacao.alocar_proporcional(votos, 1, "dhondt", True).sum() == 1


def test_sainte_lague_favorece_menores_mais_que_dhondt():
    votos = pd.Series({"A": 600, "B": 300, "C": 100})
    dhondt = alocacao.alocar_proporcional(votos, 5, "dhondt", False)
    sainte_lague = alocacao.alocar_proporcional(votos, 5, "sainte_lague", False)
    assert sainte_lague["C"] >= dhondt["C"]
    assert dhondt.sum() == sainte_lague.sum() == 5


@pytest.mark.parametrize("metodo", ["dhondt", "sainte_lague"])
def test_divisores_independem_da_ordem_de_entrada(metodo):
    votos = pd.Series({"A": 700, "B": 200, "C": 100})
    pd.testing.assert_series_equal(
        alocacao.alocar_proporcional(votos, 7, metodo, True),
        alocacao.alocar_proporcional(votos.iloc[::-1], 7, metodo, True),
    )


def test_metodo_desconhecido_gera_erro():
    with pytest.raises(ValueError, match="desconhecido"):
        alocacao.alocar_proporcional(pd.Series({"A": 100}), 1, "outro", True)


def test_cenarios_de_metodos_cruzam_cada_regra_com_cada_metodo():
    parametros = {
        "cenarios": [
            {"id": "real", "tipo": "real", "rotulo": "Real"},
            {"id": "pec_50", "tipo": "pec", "metodo": "quociente_atual", "rotulo": "PEC 50% · regra atual"},
            {"id": "distritao", "tipo": "distritao", "rotulo": "Distritão"},
        ],
        "metodos_comparados": [{"id": "quociente_atual"}, {"id": "dhondt"}],
    }
    cenarios, linhas = pipeline.montar_cenarios_metodos(parametros)
    assert [c["id"] for c in cenarios] == ["real", "pec_50__quociente_atual", "pec_50__dhondt"]
    assert [c["metodo"] for c in cenarios[1:]] == ["quociente_atual", "dhondt"]
    assert all(c["regra"] == "pec_50" for c in cenarios[1:])
    assert linhas == [{"id": "pec_50", "rotulo": "PEC 50%"}]


def test_marcar_metodos_compara_com_a_regra_atual_da_mesma_regra():
    resultados = pd.DataFrame(
        {
            "cenario": [
                "real",
                "pec_50__quociente_atual",
                "pec_50__dhondt",
                "pec_2_3__quociente_atual",
                "pec_2_3__dhondt",
            ],
            "ag": ["A"] * 5,
            "total": [9, 10, 12, 20, 19],
        }
    )
    tabela = pipeline.marcar_metodos(resultados)
    assert tabela["diferenca_vs_regra_atual"].tolist()[1:] == [0, 2, 0, -1]
    assert pd.isna(tabela["diferenca_vs_regra_atual"].iloc[0])
    assert tabela["regra"].tolist()[1:3] == ["pec_50", "pec_50"]
