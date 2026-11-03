"""Leitura e limpeza dos arquivos do TSE (seção 3 da especificação).

As funções `ler_*` fazem a leitura; as demais são puras e recebem DataFrames.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parents[1]
CHAVE_SECAO = ["CD_MUNICIPIO", "NR_ZONA", "NR_SECAO"]
VOTOS_BRANCO_NULO = {"95", "96"}
SITUACOES_ELEITO = {"ELEITO POR QP", "ELEITO POR MÉDIA"}


UFS = {
    "AC": {"codigo": "12", "nome": "Acre"},
    "AL": {"codigo": "27", "nome": "Alagoas"},
    "AM": {"codigo": "13", "nome": "Amazonas"},
    "AP": {"codigo": "16", "nome": "Amapá"},
    "BA": {"codigo": "29", "nome": "Bahia"},
    "CE": {"codigo": "23", "nome": "Ceará"},
    "DF": {"codigo": "53", "nome": "Distrito Federal"},
    "ES": {"codigo": "32", "nome": "Espírito Santo"},
    "GO": {"codigo": "52", "nome": "Goiás"},
    "MA": {"codigo": "21", "nome": "Maranhão"},
    "MG": {"codigo": "31", "nome": "Minas Gerais"},
    "MS": {"codigo": "50", "nome": "Mato Grosso do Sul"},
    "MT": {"codigo": "51", "nome": "Mato Grosso"},
    "PA": {"codigo": "15", "nome": "Pará"},
    "PB": {"codigo": "25", "nome": "Paraíba"},
    "PE": {"codigo": "26", "nome": "Pernambuco"},
    "PI": {"codigo": "22", "nome": "Piauí"},
    "PR": {"codigo": "41", "nome": "Paraná"},
    "RJ": {"codigo": "33", "nome": "Rio de Janeiro"},
    "RN": {"codigo": "24", "nome": "Rio Grande do Norte"},
    "RO": {"codigo": "11", "nome": "Rondônia"},
    "RR": {"codigo": "14", "nome": "Roraima"},
    "RS": {"codigo": "43", "nome": "Rio Grande do Sul"},
    "SC": {"codigo": "42", "nome": "Santa Catarina"},
    "SE": {"codigo": "28", "nome": "Sergipe"},
    "SP": {"codigo": "35", "nome": "São Paulo"},
    "TO": {"codigo": "17", "nome": "Tocantins"},
}


def ler_params(
    uf: str | None = None, caminho: Path = RAIZ / "config" / "params.yaml"
) -> dict:
    """Parâmetros da UF; o número de cadeiras vem dos eleitos no arquivo de candidatos."""
    with open(caminho, encoding="utf-8") as arquivo:
        params = yaml.safe_load(arquivo)
    if uf is not None:
        params["uf"] = uf
    candidatos = caminho_intermediario(params, "candidatos", params["uf"])
    if candidatos.exists():
        situacao = pd.read_parquet(candidatos, columns=["DS_SIT_TOT_TURNO"])
        params["n_cadeiras"] = int(situacao["DS_SIT_TOT_TURNO"].isin(SITUACOES_ELEITO).sum())
    return params


def caminho_fonte(params: dict, nome: str, uf: str | None = None) -> Path:
    """Arquivo original, que nunca é alterado."""
    relativo = params["arquivos"][nome].format(uf=uf)
    return (RAIZ / params["caminhos"]["fontes"] / relativo).resolve()


def caminho_intermediario(params: dict, nome: str, uf: str) -> Path:
    return RAIZ / params["caminhos"]["intermediario"] / nome / f"{uf}.parquet"


def pasta_processados(params: dict) -> Path:
    return RAIZ / params["caminhos"]["processed"] / params["uf"]


def pasta_saida(params: dict) -> Path:
    return RAIZ / params["caminhos"]["outputs"] / params["uf"]


def ler_tse(params: dict, nome: str, colunas: list[str] | None = None) -> pd.DataFrame:
    """Lê a extração da UF, toda em texto; as conversões ficam explícitas em cada etapa."""
    return pd.read_parquet(caminho_intermediario(params, nome, params["uf"]), columns=colunas)


def ler_setores(params: dict) -> gpd.GeoDataFrame:
    """Setores censitários da UF com a população residente (V0001, "Total de pessoas")."""
    setores = gpd.read_parquet(caminho_intermediario(params, "setores", params["uf"]))
    # Setores sem município são corpos d'água, sem população (Lagoa dos Patos, por exemplo).
    setores = setores[setores["CD_MUN"].notna() & setores["NM_MUN"].notna()].reset_index(drop=True)
    invalidas = ~setores.geometry.is_valid
    if invalidas.any():
        logger.info("Geometrias de setor corrigidas: %d", invalidas.sum())
        setores.loc[invalidas, "geometry"] = setores.loc[invalidas].geometry.make_valid()
    return setores


def ler_locais_complementares(params: dict) -> pd.DataFrame:
    return pd.read_parquet(caminho_intermediario(params, "locais_complementares", params["uf"]))


def ler_apelidos_municipios(
    params: dict, caminho: Path = RAIZ / "config" / "municipios_tse_ibge.csv"
) -> dict[str, str]:
    """Grafias de município que diferem entre TSE e IBGE, já sem acentos."""
    apelidos = pd.read_csv(caminho, dtype=str)
    apelidos = apelidos[apelidos["uf"] == params["uf"]]
    return dict(zip(apelidos["nome_tse"], apelidos["nome_ibge"]))


def ler_cores(caminho: Path = RAIZ / "config" / "cores_agremiacoes.csv") -> dict[str, str]:
    cores = pd.read_csv(caminho, dtype=str)
    return dict(zip(cores["agremiacao"], cores["cor"]))


def filtrar_cargo(df: pd.DataFrame, cargo: str) -> pd.DataFrame:
    """Mantém o cargo (sem diferenciar maiúsculas) e o primeiro turno."""
    return df[(df["DS_CARGO"].str.upper() == cargo.upper()) & (df["NR_TURNO"] == "1")]


def definir_agremiacao(df: pd.DataFrame) -> pd.Series:
    """Federação conta como um único partido (regra 2.1)."""
    return df["SG_FEDERACAO"].where(df["NR_FEDERACAO"] != "-1", df["SG_PARTIDO"])


def _inteiros(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    return df.assign(**{coluna: df[coluna].astype(int) for coluna in colunas})


def limpar_candidatos(candidatos: pd.DataFrame, cargo: str) -> pd.DataFrame:
    """Uma linha por candidato, com agremiação e data de nascimento."""
    df = filtrar_cargo(candidatos, cargo)
    df = df.assign(
        agremiacao=definir_agremiacao(df),
        DT_NASCIMENTO=pd.to_datetime(df["DT_NASCIMENTO"], format="%d/%m/%Y"),
    )
    colunas = [
        "SQ_CANDIDATO",
        "NR_CANDIDATO",
        "NM_URNA_CANDIDATO",
        "SG_PARTIDO",
        "agremiacao",
        "DS_SITUACAO_CANDIDATURA",
        "DT_NASCIMENTO",
        "DS_SIT_TOT_TURNO",
    ]
    return df[colunas].sort_values("SQ_CANDIDATO").reset_index(drop=True)


def calcular_votos_partido(votacao_partido: pd.DataFrame, cargo: str) -> pd.DataFrame:
    """Votos nominais e de legenda válidos por partido, com sua agremiação."""
    df = filtrar_cargo(votacao_partido, cargo)
    df = _inteiros(
        df,
        [
            "QT_VOTOS_NOMINAIS_VALIDOS",
            "QT_TOTAL_VOTOS_LEG_VALIDOS",
            "QT_VOTOS_NOM_CONVR_LEG_VALIDOS",
        ],
    )
    df = df.assign(agremiacao=definir_agremiacao(df))
    return (
        df.groupby(["NR_PARTIDO", "SG_PARTIDO", "agremiacao"], as_index=False)
        .agg(
            votos_nominais=("QT_VOTOS_NOMINAIS_VALIDOS", "sum"),
            votos_legenda=("QT_TOTAL_VOTOS_LEG_VALIDOS", "sum"),
            votos_convertidos=("QT_VOTOS_NOM_CONVR_LEG_VALIDOS", "sum"),
        )
        .sort_values("NR_PARTIDO")
        .reset_index(drop=True)
    )


def calcular_votos_ag(votos_partido: pd.DataFrame) -> pd.DataFrame:
    """Voto da agremiação = nominais válidos + legenda válidos (regra 2.1)."""
    df = votos_partido.groupby("agremiacao", as_index=False)[
        ["votos_nominais", "votos_legenda", "votos_convertidos"]
    ].sum()
    df["votos"] = df["votos_nominais"] + df["votos_legenda"]
    return df.sort_values("agremiacao").reset_index(drop=True)


def calcular_totais_validos(detalhe_munzona: pd.DataFrame, cargo: str) -> dict[str, int]:
    """Totais oficiais de votos válidos, base do quociente eleitoral."""
    df = filtrar_cargo(detalhe_munzona, cargo)
    nominais = int(df["QT_VOTOS_NOMINAIS_VALIDOS"].astype(int).sum())
    legenda = int(df["QT_TOTAL_VOTOS_LEG_VALIDOS"].astype(int).sum())
    return {"nominais": nominais, "legenda": legenda, "validos": nominais + legenda}


def calcular_votos_candidato(
    votacao_candidato: pd.DataFrame, candidatos: pd.DataFrame, cargo: str
) -> pd.DataFrame:
    """Votos válidos oficiais de cada candidato APTO."""
    df = _inteiros(filtrar_cargo(votacao_candidato, cargo), ["QT_VOTOS_NOMINAIS_VALIDOS"])
    votos = df.groupby("SQ_CANDIDATO", as_index=False).agg(
        votos=("QT_VOTOS_NOMINAIS_VALIDOS", "sum")
    )
    aptos = candidatos[candidatos["DS_SITUACAO_CANDIDATURA"] == "APTO"]
    resultado = aptos.merge(votos, on="SQ_CANDIDATO", how="left", validate="one_to_one")
    resultado["votos"] = resultado["votos"].fillna(0).astype(int)
    return resultado.sort_values("SQ_CANDIDATO").reset_index(drop=True)


def limpar_votos_secao(
    votacao_secao: pd.DataFrame,
    candidatos: pd.DataFrame,
    votos_candidato: pd.DataFrame,
    votos_partido: pd.DataFrame,
    cargo: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica os filtros de 3.2; devolve (votos nominais, votos de legenda) por seção.

    Candidato APTO pode ter zero voto válido oficial (partido com DRAP indeferido, como PMB e
    PRTB em SP em 2022): só contam os candidatos com votos válidos nos totais oficiais.
    """
    df = _inteiros(filtrar_cargo(votacao_secao, cargo), CHAVE_SECAO + ["QT_VOTOS"])
    df = df[~df["NR_VOTAVEL"].isin(VOTOS_BRANCO_NULO)]
    eh_legenda = df["NR_VOTAVEL"].str.len() == 2

    # Votos nominais: apenas candidatos com votos válidos oficiais.
    aptos = votos_candidato.loc[votos_candidato["votos"] > 0, ["SQ_CANDIDATO", "agremiacao"]]
    nominais = df[~eh_legenda].merge(aptos, on="SQ_CANDIDATO", how="inner", validate="many_to_one")
    logger.info(
        "Votos nominais removidos (INAPTO): %d",
        df.loc[~eh_legenda, "QT_VOTOS"].sum() - nominais["QT_VOTOS"].sum(),
    )

    # Votos de legenda: apenas partidos com legenda válida nos totais oficiais.
    # A legenda digitada na urna não inclui os nominais convertidos em legenda.
    legenda_digitada = votos_partido["votos_legenda"] - votos_partido["votos_convertidos"]
    partidos = votos_partido.loc[legenda_digitada > 0, ["NR_PARTIDO", "agremiacao"]]
    legenda = df[eh_legenda].merge(
        partidos, left_on="NR_VOTAVEL", right_on="NR_PARTIDO", how="inner", validate="many_to_one"
    )
    logger.info(
        "Votos de legenda removidos: %d",
        df.loc[eh_legenda, "QT_VOTOS"].sum() - legenda["QT_VOTOS"].sum(),
    )

    votos_nominais = (
        nominais.groupby(CHAVE_SECAO + ["SQ_CANDIDATO", "agremiacao"], as_index=False)
        .agg(votos=("QT_VOTOS", "sum"))
        .sort_values(CHAVE_SECAO + ["SQ_CANDIDATO"])
        .reset_index(drop=True)
    )
    votos_legenda = (
        legenda.groupby(CHAVE_SECAO + ["NR_PARTIDO", "agremiacao"], as_index=False)
        .agg(votos=("QT_VOTOS", "sum"))
        .sort_values(CHAVE_SECAO + ["NR_PARTIDO"])
        .reset_index(drop=True)
    )
    return votos_nominais, votos_legenda


def montar_secoes(
    detalhe_secao: pd.DataFrame, eleitorado_local: pd.DataFrame, cargo: str, uf: str
) -> pd.DataFrame:
    """Uma linha por seção: município, eleitorado (QT_APTOS) e local de votação atual.

    A junção nunca usa NR_LOCAL_VOTACAO, renumerado entre gerações dos arquivos (3.1).
    Coordenadas -1 viram NaN; a imputação depende dos setores e fica em distritos.py.
    """
    aptos = _inteiros(filtrar_cargo(detalhe_secao, cargo), CHAVE_SECAO + ["QT_APTOS"])
    aptos = aptos[CHAVE_SECAO + ["NM_MUNICIPIO", "QT_APTOS"]]

    locais = eleitorado_local[
        (eleitorado_local["SG_UF"] == uf)
        & (eleitorado_local["NR_TURNO"] == "1")
        & (eleitorado_local["DS_TIPO_SECAO_AGREGADA"] == "Principal")
    ]
    locais = _inteiros(locais, CHAVE_SECAO)
    locais = locais.assign(
        **{
            coluna: pd.to_numeric(locais[coluna].str.replace(",", "."))
            for coluna in ["NR_LATITUDE", "NR_LONGITUDE"]
        }
    )
    locais = locais[
        CHAVE_SECAO + ["NR_LOCAL_VOTACAO", "NM_LOCAL_VOTACAO", "NR_LATITUDE", "NR_LONGITUDE"]
    ]

    secoes = aptos.merge(locais, on=CHAVE_SECAO, how="left", validate="one_to_one")
    sem_local = secoes["NR_LOCAL_VOTACAO"].isna().sum()
    if sem_local:
        raise ValueError(f"{sem_local} seções sem local de votação.")

    sem_coordenada = secoes["NR_LATITUDE"].eq(-1) | secoes["NR_LONGITUDE"].eq(-1)
    secoes.loc[sem_coordenada, ["NR_LATITUDE", "NR_LONGITUDE"]] = float("nan")
    secoes["coordenada_ausente"] = sem_coordenada
    logger.info("Seções sem coordenada: %d", sem_coordenada.sum())
    return secoes.sort_values(CHAVE_SECAO).reset_index(drop=True)


def contar_eleitos_oficiais(candidatos: pd.DataFrame) -> pd.Series:
    """Cadeiras oficiais de 2022 por agremiação (referência da validação)."""
    eleitos = candidatos[candidatos["DS_SIT_TOT_TURNO"].isin(SITUACOES_ELEITO)]
    return eleitos.groupby("agremiacao")["SQ_CANDIDATO"].count().sort_index().rename("cadeiras")


def carregar_dados(params: dict) -> dict[str, pd.DataFrame]:
    """Etapa 1 do pipeline: produz todas as tabelas limpas."""
    cargo = params["cargo"]
    candidatos = limpar_candidatos(ler_tse(params, "candidatos"), cargo)
    votos_partido = calcular_votos_partido(ler_tse(params, "votacao_partido"), cargo)
    votos_candidato = calcular_votos_candidato(
        ler_tse(params, "votacao_candidato"), candidatos, cargo
    )
    votos_nominais, votos_legenda = limpar_votos_secao(
        ler_tse(params, "votacao_secao"), candidatos, votos_candidato, votos_partido, cargo
    )
    colunas_locais = [
        "SG_UF",
        "NR_TURNO",
        "DS_TIPO_SECAO_AGREGADA",
        *CHAVE_SECAO,
        "NR_LOCAL_VOTACAO",
        "NM_LOCAL_VOTACAO",
        "NR_LATITUDE",
        "NR_LONGITUDE",
    ]
    return {
        "candidatos": candidatos,
        "votos_partido": votos_partido,
        "votos_ag": calcular_votos_ag(votos_partido),
        "votos_candidato": votos_candidato,
        "votos_secao_candidato": votos_nominais,
        "votos_secao_legenda": votos_legenda,
        "secoes": montar_secoes(
            ler_tse(params, "detalhe_secao"),
            ler_tse(params, "eleitorado_local", colunas_locais),
            cargo,
            params["uf"],
        ),
    }
