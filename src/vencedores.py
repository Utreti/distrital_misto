"""Vencedores por distrito (regra 2.5) e distritão (regra 2.6).

Funções puras: recebem e devolvem DataFrames, sem leitura de arquivos.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _ordenar_por_votos(df: pd.DataFrame) -> pd.DataFrame:
    """Mais votos primeiro; empate vai para o mais velho e depois para SQ_CANDIDATO."""
    return (
        df.assign(_sq=df["SQ_CANDIDATO"].astype("int64"))
        .sort_values(["votos", "DT_NASCIMENTO", "_sq"], ascending=[False, True, True])
        .drop(columns="_sq")
    )


def agregar_votos_distrito(
    votos_secao: pd.DataFrame, secao_distrito: pd.DataFrame, chave_secao: list[str]
) -> pd.DataFrame:
    """Votos nominais de cada candidato em cada distrito de um mapa."""
    df = votos_secao.merge(
        secao_distrito[chave_secao + ["distrito"]],
        on=chave_secao,
        how="left",
        validate="many_to_one",
    )
    if df["distrito"].isna().any():
        raise ValueError("Há votos em seções sem distrito.")
    return df.groupby(["distrito", "SQ_CANDIDATO", "agremiacao"], as_index=False)["votos"].sum()


def apurar_vencedores(votos_distrito: pd.DataFrame, candidatos: pd.DataFrame) -> pd.DataFrame:
    """Percorre os pares (distrito, candidato) uma vez, do mais votado ao menos votado."""
    dados_candidato = candidatos[["SQ_CANDIDATO", "NM_URNA_CANDIDATO", "DT_NASCIMENTO"]]
    pares = votos_distrito.merge(
        dados_candidato, on="SQ_CANDIDATO", how="left", validate="many_to_one"
    )
    if pares["NM_URNA_CANDIDATO"].isna().any():
        raise ValueError("Há voto em candidato ausente do cadastro.")
    sem_data = pares.loc[pares["DT_NASCIMENTO"].isna(), "NM_URNA_CANDIDATO"].unique()
    if len(sem_data):
        # A data só decide empates; sem ela, o candidato fica por último no desempate.
        logger.warning("Candidatos sem data de nascimento: %s", sorted(sem_data))

    linhas = []
    distritos_definidos: set[int] = set()
    candidatos_eleitos: set[str] = set()
    for par in _ordenar_por_votos(pares).to_dict("records"):
        if par["distrito"] in distritos_definidos or par["SQ_CANDIDATO"] in candidatos_eleitos:
            continue
        distritos_definidos.add(par["distrito"])
        candidatos_eleitos.add(par["SQ_CANDIDATO"])
        linhas.append(par)

    vencedores = pd.DataFrame(linhas, columns=pares.columns).sort_values("distrito")
    if set(vencedores["distrito"]) != set(pares["distrito"]):
        raise ValueError("Há distrito sem vencedor.")
    return vencedores.reset_index(drop=True)


def apurar_distritao(votos_candidato: pd.DataFrame, n_cadeiras: int) -> pd.DataFrame:
    """Os N candidatos mais votados no estado, independentemente do partido."""
    if len(votos_candidato) < n_cadeiras:
        raise ValueError("Há menos candidatos do que cadeiras.")
    return _ordenar_por_votos(votos_candidato).head(n_cadeiras).reset_index(drop=True)
