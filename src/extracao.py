"""Extrai dos arquivos nacionais as linhas de deputado federal, em um parquet por UF.

Os CSVs nacionais têm vários GB; esta etapa roda uma vez e as demais leem só os parquets.
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import pandas as pd

import carga

logger = logging.getLogger(__name__)

TAMANHO_BLOCO = 1_000_000
LEITURA_TSE = {"sep": ";", "encoding": "cp1252", "dtype": str, "chunksize": TAMANHO_BLOCO}
NACIONAIS = [
    "candidatos",
    "votacao_partido",
    "votacao_candidato",
    "detalhe_munzona",
    "detalhe_secao",
    "eleitorado_local",
]


def _filtrar_bloco(bloco: pd.DataFrame, nome: str, cargo: str) -> pd.DataFrame:
    if nome == "eleitorado_local":
        return bloco[
            (bloco["NR_TURNO"] == "1") & (bloco["DS_TIPO_SECAO_AGREGADA"] == "Principal")
        ]
    return carga.filtrar_cargo(bloco, cargo)


def extrair_nacional(params: dict, nome: str) -> None:
    """Separa um CSV nacional por UF; só as linhas do cargo e do primeiro turno são mantidas."""
    caminho = carga.caminho_fonte(params, nome)
    blocos = []
    for bloco in pd.read_csv(caminho, **LEITURA_TSE):
        blocos.append(_filtrar_bloco(bloco, nome, params["cargo"]))
    tabela = pd.concat(blocos, ignore_index=True)
    for uf, parte in tabela.groupby("SG_UF"):
        destino = carga.caminho_intermediario(params, nome, uf)
        destino.parent.mkdir(parents=True, exist_ok=True)
        parte.to_parquet(destino, index=False)
    logger.info("%s: %d linhas em %d UFs", nome, len(tabela), tabela["SG_UF"].nunique())


def extrair_votacao_secao(params: dict, uf: str) -> None:
    """Lê o CSV de dentro do zip, sem descompactar em disco (até 11 GB por estado)."""
    caminho = carga.caminho_fonte(params, "votacao_secao", uf)
    with zipfile.ZipFile(caminho) as arquivo_zip:
        membro = next(nome for nome in arquivo_zip.namelist() if nome.endswith(f"_{uf}.csv"))
        with arquivo_zip.open(membro) as arquivo:
            blocos = [
                carga.filtrar_cargo(bloco, params["cargo"])
                for bloco in pd.read_csv(arquivo, **LEITURA_TSE)
            ]
    tabela = pd.concat(blocos, ignore_index=True)
    destino = carga.caminho_intermediario(params, "votacao_secao", uf)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_parquet(destino, index=False)
    logger.info("votacao_secao %s: %d linhas", uf, len(tabela))


def extrair_locais_complementares(params: dict, ufs: list[str]) -> None:
    """Locais de votação de outra eleição com coordenada, para completar os de 2022."""
    colunas = ["CD_MUNICIPIO", "NR_ZONA", "NR_LOCAL_VOTACAO", "NM_LOCAL_VOTACAO"]
    with zipfile.ZipFile(carga.caminho_fonte(params, "locais_complementares")) as arquivo_zip:
        for uf in ufs:
            membro = next(nome for nome in arquivo_zip.namelist() if nome.endswith(f"_{uf}.csv"))
            with arquivo_zip.open(membro) as arquivo:
                tabela = pd.read_csv(
                    arquivo,
                    sep=";",
                    encoding="cp1252",
                    dtype=str,
                    usecols=colunas + ["NR_LATITUDE", "NR_LONGITUDE"],
                )
            for coluna in ["NR_LATITUDE", "NR_LONGITUDE"]:
                tabela[coluna] = pd.to_numeric(tabela[coluna].str.replace(",", "."))
            tabela = tabela[tabela["NR_LATITUDE"].ne(-1) & tabela["NR_LONGITUDE"].ne(-1)].dropna()
            for coluna in colunas[:3]:
                tabela[coluna] = tabela[coluna].astype("int64")
            tabela = tabela.drop_duplicates().sort_values(colunas).reset_index(drop=True)
            destino = carga.caminho_intermediario(params, "locais_complementares", uf)
            destino.parent.mkdir(parents=True, exist_ok=True)
            tabela.to_parquet(destino, index=False)
            logger.info("locais_complementares %s: %d", uf, len(tabela))


def extrair_setores(params: dict, ufs: list[str]) -> None:
    """Separa a malha nacional de setores por UF, já com a população (V0001) de cada setor."""
    populacao = pd.read_csv(
        carga.caminho_fonte(params, "agregados_basico"),
        sep=";",
        encoding="latin1",
        dtype=str,
        usecols=["CD_SETOR", "v0001"],
    )
    for uf in ufs:
        filtro = f"CD_UF = '{carga.UFS[uf]['codigo']}'"
        setores = gpd.read_file(
            f"zip://{carga.caminho_fonte(params, 'setores')}", where=filtro
        )[["CD_SETOR", "CD_MUN", "NM_MUN", "geometry"]]
        setores = setores.merge(populacao, on="CD_SETOR", how="left", validate="one_to_one")
        sem_populacao = setores["v0001"].isna().sum()
        if sem_populacao:
            # O IBGE não divulga todos os setores; a população ausente vale zero na divisão.
            logger.warning("%s: %d setores sem população no arquivo básico", uf, sem_populacao)
        setores["populacao"] = pd.to_numeric(setores["v0001"], errors="coerce").fillna(0).astype(int)
        setores = setores.drop(columns="v0001").sort_values("CD_SETOR").reset_index(drop=True)
        destino: Path = carga.caminho_intermediario(params, "setores", uf)
        destino.parent.mkdir(parents=True, exist_ok=True)
        setores.to_parquet(destino, index=False)
        logger.info("setores %s: %d", uf, len(setores))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argumentos = argparse.ArgumentParser(description=__doc__)
    argumentos.add_argument("--ufs", nargs="+", required=True, help="UFs de votos e setores.")
    argumentos.add_argument("--sem-nacionais", action="store_true")
    opcoes = argumentos.parse_args()
    parametros = carga.ler_params()

    with ProcessPoolExecutor(max_workers=6) as executor:
        tarefas = []
        if not opcoes.sem_nacionais:
            tarefas += [executor.submit(extrair_nacional, parametros, n) for n in NACIONAIS]
        tarefas += [executor.submit(extrair_votacao_secao, parametros, uf) for uf in opcoes.ufs]
        for tarefa in tarefas:
            tarefa.result()
    extrair_setores(parametros, opcoes.ufs)
    extrair_locais_complementares(parametros, opcoes.ufs)
