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
