"""Testes dos mapas de base: propriedades genéricas e valores do Acre (seção 5)."""

import math

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import box

import carga
import distritos


@pytest.fixture(scope="module")
def setores(params):
    return carga.ler_setores(params)


@pytest.fixture(scope="module")
def secoes(dados, setores):
    return distritos.preparar_secoes(dados["secoes"], setores)


# Sementes vencedoras da busca completa: mantêm os testes rápidos (a busca leva cerca de 1 min).
SEMENTES_VENCEDORAS = {4: 2063, 6: 2081}


@pytest.fixture(scope="module", params=[4, 6])
def mapa(request, secoes, setores, params):
    config = params["distritos"]
    return distritos.selecionar_mapa(
        secoes,
        setores,
        request.param,
        [SEMENTES_VENCEDORAS[request.param]],
        config["tolerancia_eleitorado"],
        config["tolerancia_populacao_municipio"],
        config["max_tentativas_corte"],
    )


# Propriedades genéricas.


def test_polsby_popper_de_um_quadrado():
    assert distritos.polsby_popper(box(0, 0, 10, 10)) == pytest.approx(math.pi / 4)


def test_adjacencia_inclui_toque_por_vertice():
    gdf = gpd.GeoDataFrame(
        {"peso": [1, 1, 1]},
        geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2), box(5, 5, 6, 6)],
        crs="EPSG:5880",
    )
    grafo = distritos.construir_grafo(gdf, "peso")
    assert set(grafo.edges) == {(0, 1)}
    assert set(grafo.nodes) == {0, 1, 2}


def test_nomes_de_municipio_unem_tse_e_ibge():
    assert distritos.normalizar_nome("Mâncio Lima") == distritos.normalizar_nome("MÂNCIO LIMA")
    assert distritos.normalizar_nome("Mâncio Lima") == "MANCIO LIMA"


def test_sem_mapa_valido_gera_erro_em_vez_de_relaxar(secoes, setores):
    with pytest.raises(ValueError, match="Nenhuma das"):
        distritos.selecionar_mapa(secoes, setores, 4, [], 0.20, 0.05, 2000)


# Dados do Acre.


def test_populacao_dos_setores(setores):
    assert len(setores) == 2_215
    assert setores["populacao"].sum() == 830_018
    assert (setores["CD_MUN"] == "1200401").sum() == 669


@pytest.mark.parametrize(("n_distritos", "esperado"), [(4, 146_806), (6, 97_870)])
def test_alvo_de_eleitorado(n_distritos, esperado):
    assert round(distritos.calcular_alvo(587_222, n_distritos)) == esperado


@pytest.mark.parametrize(("n_distritos", "k"), [(4, 2), (6, 3)])
def test_so_rio_branco_e_dividido(secoes, n_distritos, k):
    eleitorado = secoes.groupby("CD_MUN")["QT_APTOS"].sum()
    alvo = distritos.calcular_alvo(int(eleitorado.sum()), n_distritos)
    divididos, na_faixa = distritos.classificar_municipios(eleitorado, alvo, 0.20)
    assert divididos.to_dict() == {"1200401": k}
    assert na_faixa == []


def test_municipio_sem_inteiro_possivel_vai_para_a_faixa():
    eleitorado = pd.Series({"cidade": 126, "capital": 1262, "media": 245, "pequena": 90})
    divididos, na_faixa = distritos.classificar_municipios(eleitorado, 100, 0.20)
    assert divididos.to_dict() == {"capital": 13, "media": 3}
    assert na_faixa == ["cidade"]


def test_repartir_distritos_soma_o_total():
    assert distritos.repartir_distritos([25.57, 0.9], 26) == [25, 1]
    assert distritos.repartir_distritos([1.4, 1.4, 1.2], 4) == [2, 1, 1]


def test_ilha_e_ligada_ao_vizinho_mais_proximo():
    gdf = gpd.GeoDataFrame(
        {"peso": [1, 1, 1], "rotulo": ["A", "B", "Ilha"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(3, 0, 4, 1)],
        crs="EPSG:5880",
    )
    grafo = distritos.construir_grafo(gdf, "peso")
    assert not nx.is_connected(grafo)
    distritos.conectar_ao_mais_proximo(grafo, gdf, {2}, "Ilha")
    assert grafo.has_edge(1, 2)
    assert nx.is_connected(grafo)


def test_pedacos_vizinhos_entre_si_tambem_ficam_ligados_ao_restante():
    """Dois pedaços próximos um do outro e longe do corpo principal da cidade."""
    gdf = gpd.GeoDataFrame(
        {"peso": [1] * 4, "rotulo": list("ABCD")},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(5, 0, 6, 1), box(6.5, 0, 7.5, 1)],
        crs="EPSG:5880",
    )
    grafo = distritos.construir_grafo(gdf, "peso")
    while not nx.is_connected(grafo):
        menor = min(nx.connected_components(grafo), key=lambda nos: (len(nos), min(nos)))
        distritos.conectar_ao_mais_proximo(grafo, gdf, set(menor), "pedaço")
    assert grafo.has_edge(2, 3)
    assert grafo.has_edge(1, 2)


def test_municipio_que_so_cabe_sozinho_vira_distrito_proprio():
    gdf = gpd.GeoDataFrame(
        {"peso": [119, 60, 60], "rotulo": ["Grande", "A", "B"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs="EPSG:5880",
    )
    grafo = distritos.construir_grafo(gdf, "peso")
    assert distritos.isolar_municipios_forcados(grafo, gdf, 100, 0.20) == [0]
    assert list(grafo.neighbors(0)) == []
    assert grafo.has_edge(1, 2)


def test_municipio_que_cabe_com_vizinho_nao_e_isolado():
    gdf = gpd.GeoDataFrame(
        {"peso": [100, 10], "rotulo": ["A", "B"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:5880",
    )
    grafo = distritos.construir_grafo(gdf, "peso")
    assert distritos.isolar_municipios_forcados(grafo, gdf, 100, 0.20) == []


def test_municipio_encurralado_e_detectado():
    gdf = gpd.GeoDataFrame(
        {"peso": [46, 87, 20], "rotulo": ["Pequena", "Grande", "Outra"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs="EPSG:5880",
    )
    grafo = distritos.construir_grafo(gdf, "peso")
    assert distritos.listar_encurralados(grafo, 100, 0.20) == [0]


def test_distribuicao_de_distritos_precisa_fechar_na_conta():
    gdf = gpd.GeoDataFrame(
        {"eleitorado": [100, 46]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:5880"
    )
    grafo = nx.Graph([(0, 1)])
    assert not distritos._distribuicao_cabe(gdf, grafo, 1, 100, 0.20)
    assert distritos._distribuicao_cabe(gdf, grafo, 2, 73, 0.20)
    assert not distritos._distribuicao_cabe(gdf, grafo, 0, 100, 0.20)


def test_grafias_do_tse_e_do_ibge_coincidem():
    assert distritos.normalizar_nome("Estrela d'Oeste") == distritos.normalizar_nome(
        "ESTRELA D OESTE"
    )
    assert distritos.normalizar_nome("Embu-Guaçu") == "EMBU GUACU"


def test_coordenadas_ausentes_sao_imputadas(dados, secoes):
    ausentes = dados["secoes"][dados["secoes"]["coordenada_ausente"]]
    assert len(ausentes) == 5
    assert ausentes["NM_LOCAL_VOTACAO"].nunique() == 2
    assert secoes[["NR_LATITUDE", "NR_LONGITUDE"]].notna().all(axis=None)


def test_locais_fora_do_municipio(secoes, setores):
    fora = distritos.listar_locais_fora_do_municipio(secoes, setores)
    assert len(fora) == 14
    assert "1200401" not in set(fora["CD_MUN"])


# Invariantes do mapa selecionado (4.2 e 5.1).


def test_toda_secao_em_um_unico_distrito(mapa, secoes):
    assert not mapa.secao_distrito.duplicated(carga.CHAVE_SECAO).any()
    assert len(mapa.secao_distrito) == len(secoes) == 2_124
    assert mapa.secao_distrito["distrito"].nunique() == mapa.n_distritos


def test_equilibrio_de_eleitorado(mapa):
    eleitorado = mapa.secao_distrito.groupby("distrito")["QT_APTOS"].sum()
    assert eleitorado.sum() == 587_222
    alvo = 587_222 / mapa.n_distritos
    assert ((eleitorado / alvo - 1).abs() <= 0.20).all()


def test_municipios_inteiros_exceto_rio_branco(mapa):
    n_por_municipio = mapa.secao_distrito.groupby("CD_MUN")["distrito"].nunique()
    assert (n_por_municipio.drop("1200401") == 1).all()
    assert n_por_municipio["1200401"] == {4: 2, 6: 3}[mapa.n_distritos]


def test_distritos_contiguos_na_geometria(mapa):
    """Cada distrito é uma peça conexa também ao olhar só para os polígonos."""
    geometrias = mapa.geometrias
    for geometria in geometrias.geometry:
        partes = list(getattr(geometria, "geoms", [geometria]))
        grafo = nx.Graph()
        grafo.add_nodes_from(range(len(partes)))
        grafo.add_edges_from(
            (i, j)
            for i, a in enumerate(partes)
            for j, b in enumerate(partes)
            if i < j and a.intersects(b)
        )
        assert nx.is_connected(grafo)


def test_selecao_e_deterministica(mapa, secoes, setores, params):
    config = params["distritos"]
    repetido = distritos.selecionar_mapa(
        secoes,
        setores,
        mapa.n_distritos,
        [SEMENTES_VENCEDORAS[mapa.n_distritos]],
        config["tolerancia_eleitorado"],
        config["tolerancia_populacao_municipio"],
        config["max_tentativas_corte"],
    )
    assert repetido.semente == mapa.semente
    pd.testing.assert_frame_equal(repetido.secao_distrito, mapa.secao_distrito)
