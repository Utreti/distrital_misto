"""Mapas distritais de base (seção 5 da especificação).

Funções puras: recebem DataFrames e devolvem DataFrames, sem leitura ou escrita de arquivos.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import random
import resource
import unicodedata
import warnings
from dataclasses import dataclass
from functools import partial

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from gerrychain import Graph
from gerrychain.partition import recursive_tree_part
from gerrychain.tree import (
    BalanceError,
    BipartitionWarning,
    PopulationBalanceError,
    ReselectException,
    bipartition_tree,
)

from carga import CHAVE_SECAO

logger = logging.getLogger(__name__)

CRS_METRICO = "EPSG:5880"
FALHAS_DE_CORTE = (BalanceError, PopulationBalanceError, ReselectException, RuntimeError)
COLUNA_PESO = "peso"
TAXA_MINIMA_POPULACAO = 0.10
# Margem mínima até o limite de eleitorado para um município virar k distritos só seus.
FOLGA_MUNICIPIO_DIVIDIDO = 0.015
N_SEMENTES_TESTE = 40


@dataclass(frozen=True)
class Mapa:
    """Mapa selecionado: seções por distrito, geometrias e estatísticas da busca."""

    n_distritos: int
    semente: int
    secao_distrito: pd.DataFrame
    geometrias: gpd.GeoDataFrame
    rotulos: dict[int, str]
    polsby_popper_medio: float
    n_validos: int
    n_falhas: int
    distritos_zoom: list[int]
    municipios_nao_inteiros: list[str]


def normalizar_nome(nome: str) -> str:
    """Remove acentos e caixa para unir nomes de município do TSE e do IBGE."""
    sem_acento = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    return " ".join(sem_acento.upper().replace("'", " ").replace("-", " ").split())


def calcular_alvo(eleitorado_total: int, n_distritos: int) -> float:
    return eleitorado_total / n_distritos


def polsby_popper(geometria: shapely.Geometry) -> float:
    """Compacidade 4π·área/perímetro²; a geometria precisa estar em projeção métrica."""
    return 4 * math.pi * geometria.area / geometria.length**2


def para_metrico(setores: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reprojeta e corrige os polígonos que a reprojeção deixa inválidos."""
    metricos = setores.to_crs(CRS_METRICO)
    invalidas = ~metricos.geometry.is_valid
    if invalidas.any():
        metricos.loc[invalidas, "geometry"] = metricos.loc[invalidas].geometry.make_valid()
    return metricos


def _completar_com_outra_eleicao(
    df: pd.DataFrame, locais: pd.DataFrame, setores: gpd.GeoDataFrame
) -> int:
    """Preenche coordenadas com o mesmo local em outra eleição; devolve as seções preenchidas.

    Procura pela chave do local, depois pelo nome dentro do município e, se os locais trouxerem
    a seção, pela mesma seção. O ponto só vale se cair no município do TSE: a numeração de um
    local pode ter sido reaproveitada em outro lugar.
    """
    colunas = ["NR_LONGITUDE", "NR_LATITUDE"]
    chave = ["CD_MUNICIPIO", "NR_ZONA", "NR_LOCAL_VOTACAO"]
    nome = ["CD_MUNICIPIO", "NM_LOCAL_VOTACAO"]
    secao = ["CD_MUNICIPIO", "NR_ZONA", "NR_SECAO"]
    buscas = (chave, nome, secao) if "NR_SECAO" in locais else (chave, nome)
    preenchidas = 0
    for campos in buscas:
        ausentes = df.loc[df["NR_LATITUDE"].isna(), campos + ["CD_MUN"]].astype({"CD_MUNICIPIO": "int64"})
        if campos is not nome:
            ausentes = ausentes.astype({coluna: "int64" for coluna in campos})
        unicos = locais.dropna(subset=campos).astype({coluna: "int64" for coluna in campos if coluna != "NM_LOCAL_VOTACAO"})
        unicos = unicos.drop_duplicates(campos)[campos + colunas]
        achados = ausentes.reset_index().merge(unicos, on=campos).set_index("index")
        if achados.empty:
            continue
        pontos = _pontos_das_secoes(achados, setores.crs)
        dentro = pontos.sjoin(
            setores[["CD_MUN", "geometry"]].rename(columns={"CD_MUN": "CD_MUN_SETOR"}),
            predicate="within",
        )
        validos = dentro.index[dentro["CD_MUN"] == dentro["CD_MUN_SETOR"]].unique()
        df.loc[validos, colunas] = achados.loc[validos, colunas]
        preenchidas += len(validos)
    return preenchidas


def preparar_secoes(
    secoes: pd.DataFrame,
    setores: gpd.GeoDataFrame,
    apelidos: dict[str, str] | None = None,
    locais_complementares: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Adiciona o código IBGE do município e imputa as coordenadas ausentes.

    `apelidos` traduz grafias do TSE (já normalizadas) para as do IBGE. Sem coordenada não há
    como medir o local mais próximo. Usa-se, nesta ordem: o mesmo local em outra eleição, a
    mediana dos locais da mesma zona eleitoral, que em cidades grandes é bem mais precisa, e o
    centroide do município (3.3).
    """
    apelidos = apelidos or {}
    codigos = dict(zip(setores["NM_MUN"].map(normalizar_nome), setores["CD_MUN"]))
    df = secoes.copy()
    nomes = df["NM_MUNICIPIO"].map(normalizar_nome)
    df["CD_MUN"] = nomes.map(lambda nome: apelidos.get(nome, nome)).map(codigos)
    if df["CD_MUN"].isna().any():
        faltantes = sorted(df.loc[df["CD_MUN"].isna(), "NM_MUNICIPIO"].unique())
        raise ValueError(f"Municípios do TSE sem correspondência no IBGE: {faltantes}")

    colunas = ["NR_LONGITUDE", "NR_LATITUDE"]
    outra_eleicao = 0
    if locais_complementares is not None:
        outra_eleicao = _completar_com_outra_eleicao(df, locais_complementares, setores)
    medianas = df.groupby(["CD_MUNICIPIO", "NR_ZONA"])[colunas].transform("median")
    df[colunas] = df[colunas].fillna(medianas)
    sem_zona = df["NR_LONGITUDE"].isna()
    if sem_zona.any():
        municipios = setores[setores["CD_MUN"].isin(df.loc[sem_zona, "CD_MUN"])]
        centroides = para_metrico(municipios).dissolve("CD_MUN").centroid.to_crs(setores.crs)
        df.loc[sem_zona, "NR_LONGITUDE"] = df.loc[sem_zona, "CD_MUN"].map(centroides.x)
        df.loc[sem_zona, "NR_LATITUDE"] = df.loc[sem_zona, "CD_MUN"].map(centroides.y)
    ausentes = df[df["coordenada_ausente"]]
    logger.warning(
        "Coordenadas imputadas: %d seções (%d eleitores), %d pelo centroide do município, "
        "%d por outra eleição.",
        len(ausentes),
        ausentes["QT_APTOS"].sum(),
        sem_zona.sum(),
        outra_eleicao,
    )
    return df


def _pontos_das_secoes(secoes: pd.DataFrame, crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        secoes,
        geometry=gpd.points_from_xy(secoes["NR_LONGITUDE"], secoes["NR_LATITUDE"]),
        crs=crs,
    )


def listar_locais_fora_do_municipio(
    secoes: pd.DataFrame, setores: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Locais cujas coordenadas caem fora do município do TSE. O município do TSE prevalece."""
    colunas = ["CD_MUN", "NM_MUNICIPIO", "NR_ZONA", "NR_LOCAL_VOTACAO", "NM_LOCAL_VOTACAO"]
    locais = secoes[~secoes["coordenada_ausente"]].drop_duplicates(colunas)
    pontos = _pontos_das_secoes(locais[colunas + ["NR_LONGITUDE", "NR_LATITUDE"]], setores.crs)
    unidos = pontos.sjoin(
        setores[["CD_MUN", "geometry"]].rename(columns={"CD_MUN": "CD_MUN_SETOR"}),
        how="left",
        predicate="within",
    )
    unidos["dentro"] = unidos["CD_MUN"] == unidos["CD_MUN_SETOR"]
    dentro = unidos.groupby(colunas)["dentro"].transform("any")
    fora = unidos.loc[~dentro, colunas + ["CD_MUN_SETOR"]].drop_duplicates(colunas)
    for local in fora.itertuples():
        logger.info(
            "Local fora do município do TSE, mantido nele: %s, %s (setor em %s)",
            local.NM_MUNICIPIO,
            local.NM_LOCAL_VOTACAO,
            local.CD_MUN_SETOR,
        )
    return fora.sort_values(colunas).reset_index(drop=True)


def localizar_em_setores(secoes: pd.DataFrame, setores: gpd.GeoDataFrame) -> pd.Series:
    """Setor de cada seção, procurando apenas entre os setores do seu próprio município.

    O local que cai fora do município fica no setor mais próximo dele: o TSE prevalece (3.3).
    """
    pontos = _pontos_das_secoes(secoes[["NR_LONGITUDE", "NR_LATITUDE"]], setores.crs)
    unidos = pontos.sjoin(setores[["CD_SETOR", "geometry"]], how="left", predicate="within")
    # Ponto sobre a divisa de dois setores: o menor código garante o determinismo.
    primeiro = unidos.sort_values("CD_SETOR").groupby(level=0)["CD_SETOR"].first()
    resultado = primeiro.reindex(secoes.index)
    sem_setor = resultado.isna()
    if sem_setor.any():
        proximos = (
            pontos[sem_setor]
            .to_crs(CRS_METRICO)
            .sjoin_nearest(setores[["CD_SETOR", "geometry"]].to_crs(CRS_METRICO), how="left")
            .sort_values("CD_SETOR")
            .groupby(level=0)["CD_SETOR"]
            .first()
        )
        resultado.loc[proximos.index] = proximos
        logger.info("Seções fora dos setores do município, no setor mais próximo: %d", len(proximos))
    return resultado


def construir_grafo(gdf: gpd.GeoDataFrame, coluna_peso: str) -> nx.Graph:
    """Grafo de adjacência por aresta ou vértice; os nós são as posições das linhas no gdf.

    Nós inteiros evitam que a ordem dos conjuntos varie entre execuções do Python.
    """
    grafo = nx.Graph()
    for posicao, peso in enumerate(gdf[coluna_peso]):
        grafo.add_node(posicao, **{COLUNA_PESO: int(peso)})
    origens, destinos = gdf.sindex.query(gdf.geometry, predicate="intersects")
    grafo.add_edges_from(
        (int(origem), int(destino))
        for origem, destino in zip(origens, destinos)
        if origem < destino
    )
    return grafo


def conectar_ao_mais_proximo(
    grafo: nx.Graph, gdf: gpd.GeoDataFrame, nos: set[int], rotulo: str
) -> None:
    """Liga um pedaço isolado (ilha, por exemplo) à unidade vizinha mais próxima."""
    fora = [no for no in grafo.nodes if no not in nos]
    geometria = shapely.union_all(gdf.geometry.values[sorted(nos)])
    distancias = gdf.geometry.iloc[fora].distance(geometria).to_numpy()
    destino = fora[int(distancias.argmin())]
    origem = min(sorted(nos), key=lambda no: gdf.geometry.iloc[no].distance(gdf.geometry.iloc[destino]))
    grafo.add_edge(origem, destino)
    logger.warning(
        "Contiguidade: %s ligado a %s (%.1f km de distância).",
        rotulo,
        gdf["rotulo"].iloc[destino],
        distancias.min() / 1000,
    )


def classificar_municipios(
    eleitorado_municipio: pd.Series, alvo: float, tolerancia: float
) -> tuple[pd.Series, list[str]]:
    """Separa os municípios grandes demais para um único distrito.

    Devolve os divididos em k >= 2 distritos internos, com k o inteiro mais próximo que respeita
    a tolerância, e os da faixa sem inteiro possível (parcela entre 1,2 e 1,6 com ±20%), que
    entram em setores no agrupamento e podem dividir um distrito com os vizinhos. Um inteiro
    colado no limite também não serve: não sobra margem para cortar a cidade.
    """
    folga = tolerancia - FOLGA_MUNICIPIO_DIVIDIDO
    parcelas = eleitorado_municipio / alvo
    divididos = {}
    na_faixa = []
    for municipio, parcela in parcelas[parcelas > 1 + tolerancia].items():
        validos = [
            k
            for k in range(2, math.ceil(parcela) + 2)
            if 1 - folga <= parcela / k <= 1 + folga
        ]
        if validos:
            divididos[municipio] = min(validos, key=lambda k: abs(parcela / k - 1))
        else:
            na_faixa.append(municipio)
    return pd.Series(divididos, dtype="int64"), sorted(na_faixa)


def isolar_municipios_forcados(
    grafo: nx.Graph, gdf: gpd.GeoDataFrame, alvo: float, tolerancia: float
) -> list[int]:
    """Separa unidades que só cabem sozinhas: somadas a qualquer vizinha, estouram o limite.

    Não é escolha de modelagem: todo mapa válido as tem como distrito próprio, mas a árvore
    aleatória raramente as isola, e sem isso a busca falha. Remove as arestas delas no grafo.
    """
    limite_inferior = alvo * (1 - tolerancia)
    limite_superior = alvo * (1 + tolerancia)
    forcados = []
    mudou = True
    while mudou:
        mudou = False
        for no in sorted(grafo.nodes):
            vizinhos = list(grafo.neighbors(no))
            peso = grafo.nodes[no][COLUNA_PESO]
            if not vizinhos or peso < limite_inferior:
                continue
            if peso + min(grafo.nodes[v][COLUNA_PESO] for v in vizinhos) > limite_superior:
                grafo.remove_edges_from([(no, vizinho) for vizinho in vizinhos])
                forcados.append(no)
                logger.info("Distrito próprio obrigatório: %s", gdf["rotulo"].iloc[no])
                mudou = True
    return forcados


def repartir_distritos(parcelas: list[float], total: int) -> list[int]:
    """Distritos de cada grupo conexo pelo método dos maiores restos."""
    soma = sum(parcelas)
    cotas = [parcela * total / soma for parcela in parcelas]
    distritos = [math.floor(cota) for cota in cotas]
    ordem = sorted(range(len(cotas)), key=lambda i: (distritos[i] - cotas[i], i))
    for i in ordem[: total - sum(distritos)]:
        distritos[i] += 1
    return distritos


def _particionar(
    grafo: Graph, k: int, alvo: float, tolerancia: float, rng: random.Random, max_tentativas: int
) -> dict[int, int]:
    if k == 1:
        return {no: 0 for no in grafo.node_indices}
    corte = partial(bipartition_tree, max_attempts=max_tentativas)
    return recursive_tree_part(
        grafo, list(range(k)), alvo, COLUNA_PESO, tolerancia, bipartition_tree_fn=corte, rng=rng
    )


def _grupos_ordenados(atribuicao: dict[int, int]) -> list[list[int]]:
    """Grupos de nós em ordem canônica, para que o rótulo não dependa da semente."""
    grupos: dict[int, list[int]] = {}
    for no, parte in atribuicao.items():
        grupos.setdefault(parte, []).append(no)
    return sorted((sorted(nos) for nos in grupos.values()), key=lambda nos: nos[0])


@dataclass
class _Regiao:
    """Área dividida em distritos: um município grande ou um grupo conexo do restante."""

    unidades: gpd.GeoDataFrame
    grafo_nx: nx.Graph
    grafo: Graph
    k: int
    alvo: float
    tolerancia: float
    eleitorado: np.ndarray
    nome: str | None


# Estado compartilhado com os processos filhos, herdado por fork.
_REGIOES: list[_Regiao] = []
_LIMITES: tuple[float, float] = (0.0, 0.0)
_MAX_TENTATIVAS = 0
_MEMORIA_TOTAL_GB: float | None = None


def _limitar_memoria(gigabytes: float | None) -> None:
    """Limita a RAM do processo atual a `gigabytes`; ao passar disso, surge MemoryError.

    RLIMIT_DATA conta também o que o processo herdou reservado e nunca usou, que não ocupa RAM;
    por isso esse excesso entra no teto. RLIMIT_AS seria ainda pior: conta todo o espaço reservado.
    """
    if gigabytes is not None:
        with open("/proc/self/status", encoding="ascii") as arquivo:
            campos = dict(linha.split(":", 1) for linha in arquivo)
        reservado = int(campos["VmData"].split()[0]) - int(campos["VmRSS"].split()[0])
        limite = int(gigabytes * 1000**3) + max(0, reservado) * 1024
        resource.setrlimit(resource.RLIMIT_DATA, (limite, limite))


def criar_piscina(processos: int):
    """Piscina de processos filhos por fork, que dividem igualmente o teto total de memória."""
    contexto = multiprocessing.get_context("fork")
    teto = None if _MEMORIA_TOTAL_GB is None else _MEMORIA_TOTAL_GB / processos
    return contexto.Pool(processos, initializer=_limitar_memoria, initargs=(teto,))


def _avaliar_semente(semente: int) -> dict | None:
    """Gera o mapa de uma semente; devolve None em falha de corte e `media` None se inválido."""
    rng = random.Random(semente)
    grupos = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BipartitionWarning)
            for regiao in _REGIOES:
                partes = _particionar(
                    regiao.grafo, regiao.k, regiao.alvo, regiao.tolerancia, rng, _MAX_TENTATIVAS
                )
                grupos.append(_grupos_ordenados(partes))
    except FALHAS_DE_CORTE:
        return None

    # A divisão interna usa população, mas a validade é medida no eleitorado real das seções.
    for regiao, grupos_regiao in zip(_REGIOES, grupos):
        for nos in grupos_regiao:
            if not _LIMITES[0] <= regiao.eleitorado[nos].sum() <= _LIMITES[1]:
                return {"semente": semente, "media": None}
            if not nx.is_connected(regiao.grafo_nx.subgraph(nos)):
                return {"semente": semente, "media": None}

    valores = [
        polsby_popper(shapely.union_all(regiao.unidades.geometry.values[nos]))
        for regiao, grupos_regiao in zip(_REGIOES, grupos)
        for nos in grupos_regiao
    ]
    return {"semente": semente, "media": sum(valores) / len(valores), "grupos": grupos}


def listar_encurralados(grafo: nx.Graph, alvo: float, tolerancia: float) -> list[int]:
    """Unidades pequenas demais sozinhas que estouram o limite com qualquer vizinha."""
    limite_inferior = alvo * (1 - tolerancia)
    limite_superior = alvo * (1 + tolerancia)
    encurralados = []
    for no in sorted(grafo.nodes):
        vizinhos = list(grafo.neighbors(no))
        peso = grafo.nodes[no][COLUNA_PESO]
        if vizinhos and peso < limite_inferior:
            if all(peso + grafo.nodes[v][COLUNA_PESO] > limite_superior for v in vizinhos):
                encurralados.append(no)
    return encurralados


def _montar_grafo_restante(
    unidades: gpd.GeoDataFrame, alvo: float, tolerancia: float
) -> nx.Graph:
    grafo = construir_grafo(unidades, "eleitorado")
    isolar_municipios_forcados(grafo, unidades, alvo, tolerancia)
    return grafo


def _distribuicao_cabe(
    unidades: gpd.GeoDataFrame, grafo: nx.Graph, k_restante: int, alvo: float, tolerancia: float
) -> bool:
    """Os grupos conexos do restante comportam exatamente `k_restante` distritos?"""
    if unidades.empty:
        return k_restante == 0
    if k_restante < 1:
        return False
    limite_inferior = alvo * (1 - tolerancia)
    somas = [
        int(unidades["eleitorado"].iloc[sorted(nos)].sum())
        for nos in nx.connected_components(grafo)
    ]
    # Pedaços pequenos serão ligados ao vizinho mais próximo; o maior grupo os representa.
    grandes = [soma for soma in somas if soma >= limite_inferior]
    if not grandes:
        return False
    grandes[grandes.index(max(grandes))] += sum(soma for soma in somas if soma < limite_inferior)
    parcelas = [soma / alvo for soma in grandes]
    distritos = repartir_distritos(parcelas, k_restante)
    return all(
        k >= 1 and 1 - tolerancia <= parcela / k <= 1 + tolerancia
        for parcela, k in zip(parcelas, distritos)
    )


def _montar_regioes(
    secoes: pd.DataFrame,
    setores_metricos: gpd.GeoDataFrame,
    n_distritos: int,
    alvo: float,
    tolerancia_eleitorado: float,
    tolerancia_populacao: float,
    extras_em_setores: tuple[str, ...] = (),
) -> tuple[list[_Regiao], pd.Series, pd.Series, list[str]]:
    """Monta as regiões a dividir; devolve também a unidade de cada seção e os municípios.

    `extras_em_setores` lista municípios que, embora caibam em um distrito, precisam ser
    divididos porque um vizinho não teria nenhum grupo válido com eles inteiros.
    """
    eleitorado_municipio = secoes.groupby("CD_MUN")["QT_APTOS"].sum()
    divididos, na_faixa = classificar_municipios(
        eleitorado_municipio, alvo, tolerancia_eleitorado
    )
    na_faixa = sorted({*na_faixa, *extras_em_setores})
    em_setores = sorted([*divididos.index, *na_faixa])

    unidade_da_secao = secoes["CD_MUN"].copy()
    for municipio in em_setores:
        filtro = secoes["CD_MUN"] == municipio
        setores_municipio = setores_metricos[setores_metricos["CD_MUN"] == municipio]
        unidade_da_secao[filtro] = localizar_em_setores(
            secoes[filtro], setores_municipio.to_crs("EPSG:4674")
        )
    aptos_por_unidade = secoes["QT_APTOS"].groupby(unidade_da_secao).sum()
    nao_inteiros = list(divididos.index) + na_faixa

    # Unidades do restante: municípios inteiros e, na faixa sem inteiro possível, setores.
    resto = setores_metricos[~setores_metricos["CD_MUN"].isin(divididos.index)]
    unidades = gpd.GeoDataFrame()
    grafo_nx = nx.Graph()
    agregados: dict[str, list[gpd.GeoDataFrame]] = {municipio: [] for municipio in divididos.index}
    geometrias_divididos = {
        municipio: shapely.union_all(
            setores_metricos.loc[setores_metricos["CD_MUN"] == municipio, "geometry"].values
        )
        for municipio in sorted(divididos.index)
    }

    def calcular_ks() -> dict[str, int] | None:
        """Distritos de cada município dividido, somados os municípios agregados a ele."""
        ks = {}
        for municipio in sorted(divididos.index):
            extras = sum(int(u["eleitorado"].iloc[0]) for u in agregados[municipio])
            total = pd.Series({municipio: int(eleitorado_municipio[municipio]) + extras})
            por_parcela, _ = classificar_municipios(total, alvo, tolerancia_eleitorado)
            if por_parcela.empty:
                return None
            ks[municipio] = int(por_parcela.iloc[0])
        return ks

    def tocados_por(geometria: shapely.Geometry) -> list[str]:
        return [m for m, g in geometrias_divididos.items() if geometria.intersects(g)]

    def agregar(nos: list[int], municipio: str, motivo: str) -> None:
        for no in nos:
            agregados[municipio].append(unidades.iloc[[no]])
        logger.warning(
            "%s: %s dividirá distrito com %s.",
            motivo,
            ", ".join(unidades["rotulo"].iloc[nos]),
            municipio,
        )

    if not resto.empty:
        populacao = resto.groupby("CD_MUN")["populacao"].sum()
        inteiros = (
            resto[~resto["CD_MUN"].isin(na_faixa)]
            .dissolve("CD_MUN", as_index=False)[["CD_MUN", "NM_MUN", "geometry"]]
            .assign(
                unidade=lambda df: df["CD_MUN"],
                rotulo=lambda df: df["NM_MUN"],
                populacao=lambda df: df["CD_MUN"].map(populacao),
            )
        )
        em_faixa = (
            resto[resto["CD_MUN"].isin(na_faixa)]
            .assign(unidade=lambda df: df["CD_SETOR"], rotulo=lambda df: df["NM_MUN"] + " (parte)")[
                ["CD_MUN", "NM_MUN", "geometry", "unidade", "rotulo", "populacao"]
            ]
        )
        unidades = pd.concat([inteiros, em_faixa], ignore_index=True)
        unidades = gpd.GeoDataFrame(unidades, geometry="geometry", crs=setores_metricos.crs)
        unidades = unidades.sort_values("unidade").reset_index(drop=True)
        unidades["eleitorado"] = (
            unidades["unidade"].map(aptos_por_unidade).fillna(0).astype("int64")
        )
        grafo_nx = _montar_grafo_restante(unidades, alvo, tolerancia_eleitorado)

        # Sem o município dividido, um vizinho pode ficar sem nenhum grupo válido: como continua
        # inteiro, ele passa a dividir um distrito com o município dividido que toca.
        while True:
            mover: set[int] = set()
            for no in listar_encurralados(grafo_nx, alvo, tolerancia_eleitorado):
                tocados = tocados_por(unidades.geometry.iloc[no])
                if tocados:
                    agregar([no], tocados[0], "Encurralado")
                    mover.add(no)
                    continue
                # Sem município dividido por perto, o menor vizinho precisa ser dividido.
                vizinhos = [v for v in grafo_nx.neighbors(no) if unidades["rotulo"].iloc[v][-7:] != "(parte)"]
                if not vizinhos:
                    raise ValueError(
                        f"{unidades['rotulo'].iloc[no]} não cabe em nenhum grupo válido."
                    )
                menor = min(vizinhos, key=lambda v: (unidades["eleitorado"].iloc[v], v))
                logger.warning(
                    "Vizinho dividido: %s, para %s caber em um distrito.",
                    unidades["rotulo"].iloc[menor],
                    unidades["rotulo"].iloc[no],
                )
                return _montar_regioes(
                    secoes,
                    setores_metricos,
                    n_distritos,
                    alvo,
                    tolerancia_eleitorado,
                    tolerancia_populacao,
                    (*extras_em_setores, unidades["CD_MUN"].iloc[menor]),
                )

            # Pedaço isolado pelo município dividido, pequeno demais, junta-se a ele.
            if not mover:
                for nos in nx.connected_components(grafo_nx):
                    nos = sorted(nos)
                    if unidades["eleitorado"].iloc[nos].sum() >= alvo * (1 - tolerancia_eleitorado):
                        continue
                    tocados = tocados_por(
                        shapely.union_all(unidades.geometry.values[nos])
                    ) if geometrias_divididos else []
                    if tocados:
                        agregar(nos, tocados[0], "Isolado")
                        mover.update(nos)

            # Se os distritos não fecham na conta, municípios vizinhos passam à região dividida.
            if not mover and geometrias_divididos:
                ks = calcular_ks()
                k_restante = n_distritos - sum(ks.values()) if ks else -1
                if not _distribuicao_cabe(
                    unidades, grafo_nx, k_restante, alvo, tolerancia_eleitorado
                ):
                    candidatos = [
                        no
                        for no in np.argsort(unidades["eleitorado"].to_numpy(), kind="stable")
                        if unidades["rotulo"].iloc[no][-7:] != "(parte)"
                        and tocados_por(unidades.geometry.iloc[no])
                    ]
                    if not candidatos:
                        break
                    no = int(candidatos[0])
                    agregar([no], tocados_por(unidades.geometry.iloc[no])[0], "Ajuste de distritos")
                    mover.add(no)

            if not mover:
                break
            unidades = unidades.drop(unidades.index[sorted(mover)]).reset_index(drop=True)
            if unidades.empty:
                grafo_nx = nx.Graph()
                break
            grafo_nx = _montar_grafo_restante(unidades, alvo, tolerancia_eleitorado)

    ks = calcular_ks() or {}
    regioes = []
    total_divididos = 0
    for municipio in sorted(divididos.index):
        setores_municipio = (
            setores_metricos[setores_metricos["CD_MUN"] == municipio]
            .assign(unidade=lambda df: df["CD_SETOR"], rotulo=lambda df: df["NM_MUN"])
        )
        pedacos = [setores_municipio, *agregados[municipio]]
        partes = gpd.GeoDataFrame(
            pd.concat(pedacos, ignore_index=True), geometry="geometry", crs=setores_metricos.crs
        )
        eleitorado = partes["unidade"].map(aptos_por_unidade).fillna(0).to_numpy(dtype="int64")
        if municipio not in ks:
            raise ValueError(f"A região de {municipio} não comporta um número inteiro de distritos.")
        k = ks[municipio]
        total_divididos += k
        grafo_cidade = construir_grafo(partes, "populacao")
        # Refaz os componentes a cada ligação: dois pedaços podem ser o vizinho mais próximo um
        # do outro, e ligá-los entre si não os liga ao resto da cidade.
        while not nx.is_connected(grafo_cidade):
            conectar_ao_mais_proximo(
                grafo_cidade,
                partes,
                set(min(nx.connected_components(grafo_cidade), key=lambda nos: (len(nos), min(nos)))),
                f"setores isolados de {partes['NM_MUN'].iloc[0]}",
            )
        regioes.append(
            _Regiao(
                unidades=partes,
                grafo_nx=grafo_cidade,
                grafo=Graph.from_networkx(grafo_cidade.copy()),
                k=k,
                alvo=partes["populacao"].sum() / k,
                tolerancia=tolerancia_populacao,
                eleitorado=eleitorado,
                nome=partes["NM_MUN"].iloc[0],
            )
        )

    k_restante = n_distritos - total_divididos
    if resto.empty or unidades.empty:
        if k_restante:
            raise ValueError("Sobraram distritos sem municípios para agrupar.")
        return regioes, unidade_da_secao, aptos_por_unidade, nao_inteiros

    # Pedaço pequeno demais para formar um distrito é ligado ao vizinho mais próximo.
    componentes = sorted(nx.connected_components(grafo_nx), key=len)
    for nos in componentes[:-1]:
        if unidades["eleitorado"].iloc[sorted(nos)].sum() < alvo * (1 - tolerancia_eleitorado):
            nomes = ", ".join(unidades["rotulo"].iloc[sorted(nos)].unique())
            conectar_ao_mais_proximo(grafo_nx, unidades, set(nos), nomes)

    componentes = sorted(
        (sorted(nos) for nos in nx.connected_components(grafo_nx)), key=lambda nos: nos[0]
    )
    parcelas = [unidades["eleitorado"].iloc[nos].sum() / alvo for nos in componentes]
    distritos = repartir_distritos(parcelas, k_restante)
    for nos, parcela, k in zip(componentes, parcelas, distritos):
        if k < 1 or not 1 - tolerancia_eleitorado <= parcela / k <= 1 + tolerancia_eleitorado:
            raise ValueError(
                f"Grupo de municípios com parcela {parcela:.2f} não comporta {k} distritos."
            )
        parte = unidades.iloc[nos].reset_index(drop=True)
        sub_nx = nx.convert_node_labels_to_integers(grafo_nx.subgraph(nos).copy(), ordering="sorted")
        regioes.append(
            _Regiao(
                unidades=parte,
                grafo_nx=sub_nx,
                grafo=Graph.from_networkx(sub_nx.copy()),
                k=k,
                alvo=alvo,
                tolerancia=tolerancia_eleitorado,
                eleitorado=parte["eleitorado"].to_numpy(dtype="int64"),
                nome=None,
            )
        )
    return regioes, unidade_da_secao, aptos_por_unidade, nao_inteiros


def _testar_regiao(argumentos: tuple[int, int]) -> bool:
    """A partição desta região, sozinha, já respeita os limites de eleitorado?"""
    indice, semente = argumentos
    regiao = _REGIOES[indice]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BipartitionWarning)
            partes = _particionar(
                regiao.grafo, regiao.k, regiao.alvo, regiao.tolerancia,
                random.Random(semente), _MAX_TENTATIVAS,
            )
    except FALHAS_DE_CORTE:
        return False
    return all(
        _LIMITES[0] <= regiao.eleitorado[nos].sum() <= _LIMITES[1]
        for nos in _grupos_ordenados(partes)
    )


def ajustar_pesos_das_regioes(
    regioes: list[_Regiao], sementes: list[int], processos: int
) -> None:
    """Troca população por eleitorado no município em que a população quase nunca dá mapa válido.

    O eleitorado de cada setor segue os locais de votação, que nem sempre acompanham onde as
    pessoas moram: partes com a mesma população podem estourar os limites de eleitorado.
    """
    for indice, regiao in enumerate(regioes):
        if regiao.nome is None or regiao.k < 2:
            continue
        trabalhos = [(indice, semente) for semente in sementes]
        if processos > 1:
            with criar_piscina(processos) as piscina:
                validos = sum(piscina.map(_testar_regiao, trabalhos, chunksize=1))
        else:
            validos = sum(map(_testar_regiao, trabalhos))
        if validos / len(trabalhos) >= TAXA_MINIMA_POPULACAO:
            continue
        media = regiao.eleitorado.sum() / regiao.k
        banda = min(media / _LIMITES[0] - 1, _LIMITES[1] / media - 1, 1 - _LIMITES[0] / media)
        grafo_nx = regiao.grafo_nx.copy()
        for no in grafo_nx.nodes:
            grafo_nx.nodes[no][COLUNA_PESO] = int(regiao.eleitorado[no])
        regiao.grafo_nx = grafo_nx
        regiao.grafo = Graph.from_networkx(grafo_nx.copy())
        regiao.alvo = media
        regiao.tolerancia = banda
        logger.warning(
            "Peso por eleitorado: %s (%d de %d testes válidos por população; tolerância %.3f).",
            regiao.nome,
            validos,
            len(trabalhos),
            banda,
        )


def selecionar_mapa(
    secoes: pd.DataFrame,
    setores: gpd.GeoDataFrame,
    n_distritos: int,
    sementes: list[int],
    tolerancia_eleitorado: float,
    tolerancia_populacao: float,
    max_tentativas: int,
    processos: int = 1,
    parar_na_primeira: bool = False,
    memoria_total_gb: float | None = None,
) -> Mapa:
    """Executa o algoritmo 5.2 para cada semente e fica com o mapa válido mais compacto.

    `secoes` vem de `preparar_secoes`; `setores` traz CD_SETOR, CD_MUN, NM_MUN e `populacao`.
    """
    global _REGIOES, _LIMITES, _MAX_TENTATIVAS, _MEMORIA_TOTAL_GB

    _MEMORIA_TOTAL_GB = memoria_total_gb
    secoes = secoes.reset_index(drop=True)
    alvo = calcular_alvo(int(secoes["QT_APTOS"].sum()), n_distritos)
    setores_metricos = para_metrico(setores).sort_values("CD_SETOR")
    regioes, unidade_da_secao, _, nao_inteiros = _montar_regioes(
        secoes, setores_metricos, n_distritos, alvo, tolerancia_eleitorado, tolerancia_populacao
    )
    _REGIOES = regioes
    _LIMITES = (alvo * (1 - tolerancia_eleitorado), alvo * (1 + tolerancia_eleitorado))
    _MAX_TENTATIVAS = max_tentativas
    ajustar_pesos_das_regioes(regioes, list(range(N_SEMENTES_TESTE)), processos)

    # imap preserva a ordem das sementes, então a parada antecipada também é determinística.
    avaliacoes = []
    if processos > 1 and len(sementes) > 1:
        with criar_piscina(processos) as piscina:
            for avaliacao in piscina.imap(_avaliar_semente, sementes, chunksize=1):
                avaliacoes.append(avaliacao)
                if parar_na_primeira and avaliacao and avaliacao["media"] is not None:
                    break
    else:
        for semente in sementes:
            avaliacoes.append(_avaliar_semente(semente))
            if parar_na_primeira and avaliacoes[-1] and avaliacoes[-1]["media"] is not None:
                break

    n_falhas = sum(avaliacao is None for avaliacao in avaliacoes)
    validas = [a for a in avaliacoes if a is not None and a["media"] is not None]
    if not validas:
        raise ValueError(
            f"Nenhuma das {len(sementes)} sementes gerou mapa válido para {n_distritos} distritos "
            f"({n_falhas} falhas de corte). A tolerância não é relaxada automaticamente."
        )
    # Empate de compacidade fica com a primeira semente da lista.
    melhor = max(validas, key=lambda avaliacao: avaliacao["media"])

    distrito_da_unidade: dict[str, int] = {}
    rotulos: dict[int, str] = {}
    geometrias = []
    distritos_zoom: list[int] = []
    proximo = 1
    for regiao, grupos in zip(regioes, melhor["grupos"]):
        for parte, nos in enumerate(grupos, start=1):
            for unidade in regiao.unidades["unidade"].iloc[nos]:
                distrito_da_unidade[unidade] = proximo
            if regiao.nome is not None:
                outros = sorted(set(regiao.unidades["rotulo"].iloc[nos]) - {regiao.nome})
                rotulos[proximo] = " + ".join([f"{regiao.nome} – parte {parte}", *outros])
            else:
                rotulos[proximo] = ", ".join(sorted(regiao.unidades["rotulo"].iloc[nos].unique()))
            geometrias.append(shapely.union_all(regiao.unidades.geometry.values[nos]))
            proximo += 1

    maior = max((r for r in regioes if r.nome is not None), key=lambda r: r.k, default=None)
    if maior is not None:
        distritos_zoom = [
            distrito for distrito, rotulo in rotulos.items() if rotulo.startswith(f"{maior.nome} – ")
        ]

    secao_distrito = secoes[CHAVE_SECAO + ["CD_MUN", "NM_MUNICIPIO", "QT_APTOS"]].copy()
    secao_distrito["distrito"] = unidade_da_secao.map(distrito_da_unidade)
    if secao_distrito["distrito"].isna().any():
        raise ValueError("Há seção cuja unidade ficou sem distrito.")
    secao_distrito["distrito"] = secao_distrito["distrito"].astype(int)

    mapa = Mapa(
        n_distritos=n_distritos,
        semente=melhor["semente"],
        secao_distrito=secao_distrito.sort_values(CHAVE_SECAO).reset_index(drop=True),
        geometrias=gpd.GeoDataFrame(
            {"distrito": range(1, n_distritos + 1)}, geometry=geometrias, crs=CRS_METRICO
        ),
        rotulos=rotulos,
        polsby_popper_medio=melhor["media"],
        n_validos=len(validas),
        n_falhas=n_falhas,
        distritos_zoom=distritos_zoom,
        municipios_nao_inteiros=sorted(nao_inteiros),
    )
    verificar_invariantes(mapa, secoes, tolerancia_eleitorado)
    return mapa


def verificar_invariantes(mapa: Mapa, secoes: pd.DataFrame, tolerancia: float) -> None:
    """Invariantes de 4.2: seção em um único distrito e municípios inteiros quando cabem."""
    df = mapa.secao_distrito
    if df.duplicated(CHAVE_SECAO).any() or len(df) != len(secoes):
        raise ValueError("Há seção ausente ou em mais de um distrito.")
    if df["distrito"].nunique() != mapa.n_distritos:
        raise ValueError("Há distrito sem seções.")

    distritos_por_municipio = df.groupby("CD_MUN")["distrito"].nunique()
    if (distritos_por_municipio.drop(mapa.municipios_nao_inteiros) != 1).any():
        raise ValueError("Município que deveria ficar inteiro foi dividido.")

    alvo = calcular_alvo(int(df["QT_APTOS"].sum()), mapa.n_distritos)
    desvios = (df.groupby("distrito")["QT_APTOS"].sum() / alvo - 1).abs()
    if (desvios > tolerancia).any():
        raise ValueError(f"Distrito fora da tolerância de eleitorado: {desvios.to_dict()}")


def resumir_mapa(mapa: Mapa) -> pd.DataFrame:
    """Uma linha por distrito; vencedor e votos são acrescentados na etapa 5."""
    eleitorado = mapa.secao_distrito.groupby("distrito")["QT_APTOS"].sum()
    alvo = calcular_alvo(int(eleitorado.sum()), mapa.n_distritos)
    resumo = pd.DataFrame(
        {
            "mapa": mapa.n_distritos,
            "distrito": eleitorado.index,
            "municipios": [mapa.rotulos[distrito] for distrito in eleitorado.index],
            "eleitorado": eleitorado.values,
            "desvio_alvo_pct": ((eleitorado / alvo - 1) * 100).round(2).values,
            "polsby_popper": mapa.geometrias.set_index("distrito")
            .geometry.map(polsby_popper)
            .reindex(eleitorado.index)
            .round(4)
            .values,
            "semente": mapa.semente,
        }
    )
    return resumo
