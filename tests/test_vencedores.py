"""Testes das regras de vencedor por distrito (2.5) e do distritão (2.6)."""

import pandas as pd

import carga
import pipeline
import vencedores


def _candidatos(*linhas):
    return pd.DataFrame(linhas, columns=["SQ_CANDIDATO", "NM_URNA_CANDIDATO", "DT_NASCIMENTO"]).assign(
        DT_NASCIMENTO=lambda df: pd.to_datetime(df["DT_NASCIMENTO"])
    )


def _votos(*linhas):
    return pd.DataFrame(linhas, columns=["distrito", "SQ_CANDIDATO", "agremiacao", "votos"])


def test_candidato_vence_apenas_seu_distrito_mais_forte():
    candidatos = _candidatos(("1", "Ana", "1970-01-01"), ("2", "Bia", "1970-01-01"))
    votos = _votos((1, "1", "X", 100), (2, "1", "X", 90), (1, "2", "Y", 50), (2, "2", "Y", 40))
    resultado = vencedores.apurar_vencedores(votos, candidatos)
    assert resultado.set_index("distrito")["SQ_CANDIDATO"].to_dict() == {1: "1", 2: "2"}


def test_empate_vai_para_o_mais_velho():
    candidatos = _candidatos(("1", "Nova", "1990-01-01"), ("2", "Velha", "1950-01-01"))
    votos = _votos((1, "1", "X", 100), (1, "2", "Y", 100))
    assert vencedores.apurar_vencedores(votos, candidatos)["SQ_CANDIDATO"].tolist() == ["2"]


def test_sem_data_de_nascimento_fica_por_ultimo_no_desempate():
    candidatos = pd.DataFrame(
        {
            "SQ_CANDIDATO": ["1", "2"],
            "NM_URNA_CANDIDATO": ["Sem data", "Com data"],
            "DT_NASCIMENTO": pd.to_datetime([None, "1990-01-01"]),
        }
    )
    votos = _votos((1, "1", "X", 100), (1, "2", "Y", 100))
    assert vencedores.apurar_vencedores(votos, candidatos)["SQ_CANDIDATO"].tolist() == ["2"]


def test_empate_total_usa_sq_candidato_numerico():
    candidatos = _candidatos(("9", "A", "1970-01-01"), ("10", "B", "1970-01-01"))
    votos = _votos((1, "10", "X", 100), (1, "9", "Y", 100))
    assert vencedores.apurar_vencedores(votos, candidatos)["SQ_CANDIDATO"].tolist() == ["9"]


def test_vencedores_independem_da_ordem_de_entrada():
    candidatos = _candidatos(("1", "Ana", "1970-01-01"), ("2", "Bia", "1960-01-01"))
    votos = _votos((1, "1", "X", 100), (2, "1", "X", 90), (1, "2", "Y", 100), (2, "2", "Y", 40))
    invertidos = votos.iloc[::-1]
    pd.testing.assert_frame_equal(
        vencedores.apurar_vencedores(votos, candidatos),
        vencedores.apurar_vencedores(invertidos, candidatos),
    )


def test_distritao_pega_os_mais_votados_com_desempate_por_idade():
    votos = pd.DataFrame(
        {
            "SQ_CANDIDATO": ["1", "2", "3"],
            "agremiacao": ["X", "Y", "Z"],
            "votos": [100, 50, 50],
            "DT_NASCIMENTO": pd.to_datetime(["1980-01-01", "1990-01-01", "1950-01-01"]),
        }
    )
    assert vencedores.apurar_distritao(votos, 2)["SQ_CANDIDATO"].tolist() == ["1", "3"]


def test_agregacao_soma_secoes_do_mesmo_distrito():
    votos_secao = pd.DataFrame(
        {
            "CD_MUNICIPIO": [1, 1, 2],
            "NR_ZONA": [1, 1, 1],
            "NR_SECAO": [1, 2, 1],
            "SQ_CANDIDATO": ["1", "1", "1"],
            "agremiacao": ["X", "X", "X"],
            "votos": [10, 20, 40],
        }
    )
    secao_distrito = pd.DataFrame(
        {"CD_MUNICIPIO": [1, 1, 2], "NR_ZONA": [1, 1, 1], "NR_SECAO": [1, 2, 1], "distrito": [1, 1, 2]}
    )
    resultado = vencedores.agregar_votos_distrito(votos_secao, secao_distrito, carga.CHAVE_SECAO)
    assert resultado.set_index("distrito")["votos"].to_dict() == {1: 30, 2: 40}


def _cenarios_de_teste():
    return [
        {"id": "real", "tipo": "real"},
        {
            "id": "pec",
            "tipo": "pec",
            "proporcao_distrital": "1/2",
            "metodo": "quociente_atual",
            "barreira": True,
        },
    ]


def test_tabela_de_resultados_registra_excedente_e_variacao():
    votos = pd.Series({"A": 900, "B": 700, "C": 100})
    oficial = pd.Series({"A": 2, "B": 2})
    # C ganha um distrito sem atingir a barreira: mantém a cadeira, que é excedente.
    distritos_vencidos = pd.DataFrame({"distrito": [1, 2], "agremiacao": ["A", "C"]})
    params = {"cenarios": _cenarios_de_teste(), "n_cadeiras": 4}
    resultados = pipeline.montar_resultados(
        params, votos, oficial, {2: distritos_vencidos}, pd.DataFrame()
    )
    linha_c = resultados[(resultados["cenario"] == "pec") & (resultados["ag"] == "C")].iloc[0]
    assert (linha_c["cadeiras_E"], linha_c["excedente"], linha_c["total"]) == (0, 1, 1)
    assert linha_c["variacao_vs_real"] == 1
    assert resultados["ag"].tolist()[:3] == ["A", "B", "C"]

    excedentes = pipeline.montar_excedentes(resultados, params["cenarios"], "AC")
    assert excedentes.set_index("cenario")["total_estado"].to_dict() == {"real": 4, "pec": 5}
    assert excedentes["acima_de_70"].sum() == 0


def test_excedentes_marcam_estado_acima_de_70():
    resultados = pd.DataFrame(
        {"cenario": ["pec"], "ag": ["A"], "excedente": [2], "total": [71]}
    )
    excedentes = pipeline.montar_excedentes(resultados, [{"id": "pec"}], "SP")
    assert excedentes["acima_de_70"].tolist() == [True]


def test_toda_agremiacao_do_acre_tem_cor(votos_ag):
    assert set(votos_ag.index) <= set(carga.ler_cores())


def test_mapas_do_acre_exigidos_pelos_cenarios(params):
    assert pipeline.listar_mapas(params) == [4, 6]


def test_distritao_do_acre_elege_oito_candidatos(dados):
    eleitos = vencedores.apurar_distritao(dados["votos_candidato"], 8)
    assert len(eleitos) == 8
    assert eleitos["SQ_CANDIDATO"].is_unique
    assert eleitos["votos"].is_monotonic_decreasing
