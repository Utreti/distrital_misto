"""Testes obrigatórios da seção 4.1 sobre a carga dos dados do Acre 2022."""

import pandas as pd

import carga


def test_total_de_votos_validos(params):
    detalhe = carga.ler_tse(params, "detalhe_munzona")
    totais = carga.calcular_totais_validos(detalhe, params["cargo"])
    assert totais == {"nominais": 418_749, "legenda": 15_504, "validos": 434_253}


def test_votos_das_agremiacoes_somam_o_total_valido(votos_ag):
    assert votos_ag.sum() == 434_253


def test_federacao_conta_como_uma_agremiacao():
    df = pd.DataFrame(
        {
            "NR_FEDERACAO": ["1", "1", "-1"],
            "SG_FEDERACAO": ["PSDB/CIDADANIA", "PSDB/CIDADANIA", "#NULO"],
            "SG_PARTIDO": ["PSDB", "CIDADANIA", "MDB"],
        }
    )
    assert carga.definir_agremiacao(df).tolist() == ["PSDB/CIDADANIA", "PSDB/CIDADANIA", "MDB"]


def test_votos_de_secao_batem_com_munzona_por_candidato(dados):
    secao = dados["votos_secao_candidato"].groupby("SQ_CANDIDATO")["votos"].sum()
    oficial = dados["votos_candidato"].set_index("SQ_CANDIDATO")["votos"]
    oficial = oficial[oficial > 0].sort_index()
    pd.testing.assert_series_equal(secao.sort_index(), oficial, check_names=False)


def test_votos_de_secao_batem_com_munzona_por_agremiacao(dados, votos_ag):
    somas = pd.concat([dados["votos_secao_candidato"], dados["votos_secao_legenda"]])
    secao = somas.groupby("agremiacao")["votos"].sum().sort_index()
    pd.testing.assert_series_equal(secao, votos_ag.sort_index(), check_names=False)


def test_todas_as_secoes_com_voto_tem_local(params, dados):
    votacao = carga.filtrar_cargo(
        carga.ler_tse(params, "votacao_secao", carga.CHAVE_SECAO + ["DS_CARGO", "NR_TURNO"]),
        params["cargo"],
    )
    com_voto = votacao[carga.CHAVE_SECAO].astype(int).drop_duplicates()
    unidas = com_voto.merge(dados["secoes"], on=carga.CHAVE_SECAO, how="inner")
    assert len(com_voto) == 2_124
    assert len(unidas) == 2_124
    assert unidas["NR_LOCAL_VOTACAO"].notna().all()


def test_cadeiras_vem_dos_eleitos(params):
    assert params["n_cadeiras"] == 8
    assert carga.ler_params("SP")["n_cadeiras"] == 70


def test_eleitorado_por_secao(dados):
    assert dados["secoes"]["QT_APTOS"].sum() == 587_222


def test_votos_validos_de_quem_perdeu_o_registro_continuam_valendo():
    candidatos = pd.DataFrame(
        {
            "SQ_CANDIDATO": ["1", "2", "3"],
            "DS_SITUACAO_CANDIDATURA": ["APTO", "INAPTO", "INAPTO"],
        }
    )
    votacao = pd.DataFrame(
        {
            "DS_CARGO": "DEPUTADO FEDERAL",
            "NR_TURNO": "1",
            "SQ_CANDIDATO": ["1", "2", "3"],
            "QT_VOTOS_NOMINAIS_VALIDOS": ["10", "20", "0"],
        }
    )
    resultado = carga.calcular_votos_candidato(votacao, candidatos, "Deputado Federal")
    assert resultado.set_index("SQ_CANDIDATO")["votos"].to_dict() == {"1": 10, "2": 20}


def test_registro_que_so_repete_a_legenda_nao_conta_duas_vezes():
    comuns = {
        "DS_CARGO": "DEPUTADO FEDERAL",
        "NR_TURNO": "1",
        "NR_PARTIDO": "12",
        "SG_PARTIDO": "PDT",
        "NR_FEDERACAO": "-1",
        "SG_FEDERACAO": "#NULO#",
        "QT_VOTOS_NOM_CONVR_LEG_VALIDOS": "0",
    }
    votacao = pd.DataFrame(
        [
            {**comuns, "SQ_COLIGACAO": "1", "QT_VOTOS_NOMINAIS_VALIDOS": "50", "QT_TOTAL_VOTOS_LEG_VALIDOS": "7"},
            {**comuns, "SQ_COLIGACAO": "2", "QT_VOTOS_NOMINAIS_VALIDOS": "0", "QT_TOTAL_VOTOS_LEG_VALIDOS": "7"},
        ]
    )
    linha = carga.calcular_votos_partido(votacao, "Deputado Federal").iloc[0]
    assert (linha["votos_nominais"], linha["votos_legenda"], linha["SQ_COLIGACAO"]) == (50, 7, "1")
