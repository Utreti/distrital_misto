"""Motor de alocação proporcional e regra compensatória (regras 2.2, 2.3 e 2.7).

Funções puras: recebem e devolvem Series/valores, sem leitura de arquivos.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from fractions import Fraction

import pandas as pd

logger = logging.getLogger(__name__)

# Próximo divisor de cada método, dado o número de cadeiras que a agremiação já tem.
DIVISORES = {
    "dhondt": lambda cadeiras: cadeiras + 1,
    "sainte_lague": lambda cadeiras: 2 * cadeiras + 1,
}


@dataclass(frozen=True)
class DetalheQuociente:
    """Etapas intermediárias da regra atual, usadas na validação contra o oficial."""

    qe: int
    quociente_partidario: pd.Series
    elegiveis: tuple[str, ...]
    ordem_sobras: tuple[str, ...]
    cadeiras: pd.Series


def calcular_qe(total_votos: int, n_cadeiras: int) -> int:
    """Quociente eleitoral; fração de exatamente meio é desprezada (art. 106 do CE)."""
    return (2 * total_votos + n_cadeiras - 1) // (2 * n_cadeiras)


def _normalizar_votos(votos: pd.Series) -> pd.Series:
    """Ordena por agremiação para que o resultado independa da ordem de entrada."""
    if votos.index.has_duplicates:
        raise ValueError("Há agremiações duplicadas no vetor de votos.")
    if (votos < 0).any():
        raise ValueError("Há agremiações com votos negativos.")
    if votos.sum() <= 0:
        raise ValueError("O total de votos precisa ser positivo.")
    return votos.astype(int).sort_index()


def _proxima_agremiacao(
    votos: pd.Series, cadeiras: pd.Series, grupo: list[str], divisor=DIVISORES["dhondt"]
) -> str:
    """Maior média; empate vai para mais votos e, persistindo, para a ordem do nome (2.7)."""
    medias = {ag: Fraction(int(votos[ag]), divisor(int(cadeiras[ag]))) for ag in grupo}
    maior_media = max(medias.values())
    empatadas = [ag for ag, media in medias.items() if media == maior_media]
    mais_votos = max(int(votos[ag]) for ag in empatadas)
    finalistas = sorted(ag for ag in empatadas if int(votos[ag]) == mais_votos)
    if len(finalistas) > 1:
        logger.warning("Empate persistente resolvido pelo nome da agremiação: %s", finalistas)
    return finalistas[0]


def detalhar_quociente_atual(
    votos: pd.Series, n_cadeiras: int, barreira: bool
) -> DetalheQuociente:
    """Aplica a regra atual (CE arts. 106-109), guardando cada etapa."""
    votos = _normalizar_votos(votos)
    qe = calcular_qe(int(votos.sum()), n_cadeiras)
    if qe < 1:
        raise ValueError("Votos insuficientes para calcular o quociente eleitoral.")

    quociente_partidario = votos // qe
    if quociente_partidario.sum() > n_cadeiras:
        # Só ocorre com totais ínfimos (QE <= N/2), nunca em eleições reais.
        raise ValueError("Os quocientes partidários excedem o número de cadeiras.")

    # 5 * votos >= 4 * QE evita arredondamento em 80% do QE.
    elegiveis = [ag for ag in votos.index if 5 * int(votos[ag]) >= 4 * qe] if barreira else []
    todas = list(votos.index)

    cadeiras = quociente_partidario.copy()
    ordem_sobras = []
    for _ in range(n_cadeiras - int(cadeiras.sum())):
        # Sem agremiação elegível (ou sem barreira), a disputa é entre todas.
        ag = _proxima_agremiacao(votos, cadeiras, elegiveis or todas)
        cadeiras[ag] += 1
        ordem_sobras.append(ag)

    return DetalheQuociente(
        qe=qe,
        quociente_partidario=quociente_partidario,
        elegiveis=tuple(elegiveis),
        ordem_sobras=tuple(ordem_sobras),
        cadeiras=cadeiras,
    )


def alocar_por_divisores(
    votos: pd.Series, n_cadeiras: int, metodo: str, barreira: bool
) -> pd.Series:
    """Cada cadeira vai para a maior média; a barreira exclui quem tem menos de 80% do QE."""
    votos = _normalizar_votos(votos)
    qe = calcular_qe(int(votos.sum()), n_cadeiras)
    elegiveis = [ag for ag in votos.index if 5 * int(votos[ag]) >= 4 * qe] if barreira else []
    grupo = elegiveis or list(votos.index)
    cadeiras = pd.Series(0, index=votos.index, dtype="int64")
    for _ in range(n_cadeiras):
        cadeiras[_proxima_agremiacao(votos, cadeiras, grupo, DIVISORES[metodo])] += 1
    return cadeiras


def alocar_proporcional(
    votos: pd.Series, n_cadeiras: int, metodo: str, barreira: bool
) -> pd.Series:
    """Distribui `n_cadeiras` entre as agremiações do índice de `votos`."""
    if metodo == "quociente_atual":
        return detalhar_quociente_atual(votos, n_cadeiras, barreira).cadeiras
    if metodo in DIVISORES:
        return alocar_por_divisores(votos, n_cadeiras, metodo, barreira)
    raise ValueError(f"Método desconhecido: '{metodo}'.")


def alocar_lista_aberta(
    votos: pd.Series, votos_candidatos: pd.DataFrame, n_cadeiras: int
) -> pd.Series:
    """Regra real desde 2022, usada só para validar: inclui as exigências individuais.

    Vaga do quociente partidário exige candidato com 10% do QE; sobra exige agremiação com 80%
    do QE e candidato ainda não eleito com 20% do QE (CE art. 108 e 109). `votos_candidatos`
    traz `agremiacao` e `votos`. Sem disputantes, as sobras vão para todas, sem exigências.
    """
    votos = _normalizar_votos(votos)
    qe = calcular_qe(int(votos.sum()), n_cadeiras)
    com_10 = votos_candidatos[10 * votos_candidatos["votos"] >= qe].groupby("agremiacao").size()
    com_20 = votos_candidatos[5 * votos_candidatos["votos"] >= qe].groupby("agremiacao").size()
    com_10 = com_10.reindex(votos.index, fill_value=0)
    com_20 = com_20.reindex(votos.index, fill_value=0)

    cadeiras = pd.concat([votos // qe, com_10], axis=1).min(axis=1).astype(int)
    for _ in range(n_cadeiras - int(cadeiras.sum())):
        grupo = [
            ag
            for ag in votos.index
            if 5 * int(votos[ag]) >= 4 * qe and int(com_20[ag]) > int(cadeiras[ag])
        ]
        ag = _proxima_agremiacao(votos, cadeiras, grupo or list(votos.index))
        cadeiras[ag] += 1
    return cadeiras


def alocar_com_coligacoes(
    votos: pd.Series, votos_candidatos: pd.DataFrame, coligacoes: pd.Series, n_cadeiras: int
) -> pd.Series:
    """Regra real de 2018, usada só para validar; devolve as cadeiras de cada partido.

    As vagas são da coligação (`coligacoes` liga partido a coligação) e vão aos seus candidatos
    mais votados. Toda vaga exige candidato com 10% do QE; sobra por maior média, aberta a todas as
    coligações (CE art. 108 e 109, Lei 13.488/2017). Sem disputantes, vale só a maior média.
    `votos_candidatos` traz `agremiacao` e `votos`.
    """
    votos = _normalizar_votos(votos)
    votos_coligacao = votos.groupby(coligacoes.reindex(votos.index)).sum()
    candidatos = votos_candidatos[votos_candidatos["agremiacao"].isin(votos.index)].assign(
        coligacao=lambda df: df["agremiacao"].map(coligacoes)
    )
    qe = calcular_qe(int(votos.sum()), n_cadeiras)
    com_10 = candidatos[10 * candidatos["votos"] >= qe].groupby("coligacao").size()
    com_10 = com_10.reindex(votos_coligacao.index, fill_value=0)

    cadeiras = pd.concat([votos_coligacao // qe, com_10], axis=1).min(axis=1).astype(int)
    for _ in range(n_cadeiras - int(cadeiras.sum())):
        grupo = [c for c in votos_coligacao.index if int(com_10[c]) > int(cadeiras[c])]
        coligacao = _proxima_agremiacao(
            votos_coligacao, cadeiras, grupo or list(votos_coligacao.index)
        )
        cadeiras[coligacao] += 1

    eleitos = candidatos.sort_values("votos", ascending=False, kind="stable").groupby("coligacao").head(
        n_cadeiras
    )
    eleitos = eleitos[eleitos.groupby("coligacao").cumcount() < eleitos["coligacao"].map(cadeiras)]
    return eleitos.groupby("agremiacao").size().reindex(votos.index, fill_value=0).astype(int)


def calcular_n_distritos(n_cadeiras: int, proporcao: Fraction) -> int:
    """Número de distritos: arredonda para cima, inclusive "o número imediatamente superior"."""
    return math.ceil(n_cadeiras * proporcao)


def combinar_pec(cadeiras_e: pd.Series, distritos_d: pd.Series) -> pd.DataFrame:
    """Regra compensatória (2.3); E_i precisa ter sido calculado sobre TODAS as N cadeiras."""
    agremiacoes = cadeiras_e.index.union(distritos_d.index)
    df = pd.DataFrame(
        {
            "cadeiras_E": cadeiras_e.reindex(agremiacoes, fill_value=0).astype(int),
            "distritos_D": distritos_d.reindex(agremiacoes, fill_value=0).astype(int),
        }
    )
    df["lista"] = (df["cadeiras_E"] - df["distritos_D"]).clip(lower=0)
    df["excedente"] = (df["distritos_D"] - df["cadeiras_E"]).clip(lower=0)
    df["total"] = df[["cadeiras_E", "distritos_D"]].max(axis=1)

    # Invariantes de 4.2.
    n_cadeiras = int(df["cadeiras_E"].sum())
    if df["total"].sum() != n_cadeiras + df["excedente"].sum():
        raise ValueError("Total de cadeiras difere de N + excedentes.")
    if df["lista"].sum() + df["distritos_D"].sum() != df["total"].sum():
        raise ValueError("Lista + distritos difere do total de cadeiras.")
    return df
