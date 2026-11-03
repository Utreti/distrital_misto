"""Fase 3: conjunto de mapas válidos gerados por ReCom, com a regra PEC aplicada a cada um.

Cada região (município dividido ou grupo conexo do restante) evolui com sua própria cadeia, sob
as mesmas regras do mapa de base. Amostras de cadeias independentes medem quanto o mapa desloca
os vencedores dos distritos e as cadeiras excedentes.
"""

from __future__ import annotations

import argparse
import logging
import random
import warnings
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from gerrychain import Graph, Partition
from gerrychain.proposals import recom
from gerrychain.tree import BipartitionWarning, bipartition_tree
from gerrychain.updaters import Tally

import alocacao
import carga
import distritos
import figuras

logger = logging.getLogger(__name__)

COLUNA_APTOS = "aptos"
LIMITE_CONSTITUCIONAL = 70
QUANTIS = {"p5": 5, "mediana": 50, "p95": 95}


@dataclass
class Votos:
    """Votos nominais em formato esparso: unidade x candidato."""

    unidades: np.ndarray
    candidatos: np.ndarray
    votos: np.ndarray
    n_candidatos: int
    ranking: np.ndarray  # posição do candidato no desempate (mais velho primeiro)
    agremiacao: np.ndarray  # índice da agremiação de cada candidato
    n_agremiacoes: int


@dataclass
class Problema:
    regioes: list[distritos._Regiao]
    grupos_iniciais: list[list[list[list[int]]]]  # um mapa de partida por cadeia
    primeira_unidade: list[int]
    primeiro_distrito: list[int]
    votos: Votos
    limites: tuple[float, float]
    n_distritos: int
    max_tentativas: int
    n_burn_in: int
    passos_por_amostra: float
    semente_base: int


_PROBLEMA: Problema | None = None


def apurar_distritos(
    distrito_da_unidade: np.ndarray, votos: Votos, n_distritos: int
) -> tuple[np.ndarray, np.ndarray]:
    """Aplica a regra 2.5; devolve os distritos por agremiação e o candidato vencedor de cada um."""
    linhas = distrito_da_unidade[votos.unidades]
    chave = linhas.astype("int64") * votos.n_candidatos + votos.candidatos
    somas = np.bincount(chave, weights=votos.votos, minlength=n_distritos * votos.n_candidatos)
    pares = np.flatnonzero(somas)
    # Mais votos primeiro; empate fica com o candidato de melhor posição no desempate.
    ordem = np.lexsort((votos.ranking[pares % votos.n_candidatos], -somas[pares]))

    vencedores = np.full(n_distritos, -1, dtype="int64")
    usados: set[int] = set()
    faltam = n_distritos
    for par in pares[ordem]:
        distrito, candidato = int(par // votos.n_candidatos), int(par % votos.n_candidatos)
        if vencedores[distrito] >= 0 or candidato in usados:
            continue
        vencedores[distrito] = candidato
        usados.add(candidato)
        faltam -= 1
        if not faltam:
            break
    if faltam:
        raise ValueError("Há distrito sem vencedor.")
    return (
        np.bincount(votos.agremiacao[vencedores], minlength=votos.n_agremiacoes),
        vencedores,
    )


def _criar_particao(regiao: distritos._Regiao, grupos: list[list[int]]) -> tuple[Partition, Graph]:
    grafo_nx = regiao.grafo_nx.copy()
    for no in grafo_nx.nodes:
        grafo_nx.nodes[no][COLUNA_APTOS] = int(regiao.eleitorado[no])
    grafo = Graph.from_networkx(grafo_nx)
    atribuicao = {no: parte for parte, nos in enumerate(grupos) for no in nos}
    particao = Partition(
        grafo,
        atribuicao,
        updaters={
            distritos.COLUNA_PESO: Tally(distritos.COLUNA_PESO, alias=distritos.COLUNA_PESO),
            COLUNA_APTOS: Tally(COLUNA_APTOS, alias=COLUNA_APTOS),
        },
    )
    return particao, grafo


def _dar_passo(particao: Partition, regiao: distritos._Regiao, problema: Problema, rng) -> tuple[Partition, bool]:
    """Um passo de ReCom; proposta inválida ou sem corte mantém o mapa atual."""
    corte = partial(bipartition_tree, max_attempts=problema.max_tentativas)
    try:
        proposta = recom(
            particao,
            pop_col=distritos.COLUNA_PESO,
            pop_target=regiao.alvo,
            epsilon=regiao.tolerancia,
            node_repeats=0,
            bipartition_tree_fn=corte,
            rng=rng,
        )
    except distritos.FALHAS_DE_CORTE:
        return particao, False
    aptos = proposta[COLUNA_APTOS].values()
    if min(aptos) < problema.limites[0] or max(aptos) > problema.limites[1]:
        return particao, False
    # Como o MarkovChain do gerrychain: sem isto, cada mapa guarda toda a sua linhagem na memória.
    particao.parent = None
    return proposta, True


def _vetor_de_distritos(particoes: list[Partition | None], problema: Problema) -> np.ndarray:
    n_unidades = sum(len(regiao.unidades) for regiao in problema.regioes)
    vetor = np.zeros(n_unidades, dtype="int64")
    for indice, (regiao, particao) in enumerate(zip(problema.regioes, particoes)):
        base = problema.primeira_unidade[indice]
        primeiro = problema.primeiro_distrito[indice]
        if particao is None:
            vetor[base : base + len(regiao.unidades)] = primeiro
            continue
        for no, parte in particao.assignment.items():
            vetor[base + no] = primeiro + parte
    return vetor


def _rodar_cadeia(argumentos: tuple[int, int]) -> dict:
    """Roda uma cadeia e devolve os distritos por agremiação em cada amostra."""
    indice_cadeia, n_amostras = argumentos
    problema = _PROBLEMA
    rng = random.Random(problema.semente_base + indice_cadeia)
    particoes: list[Partition | None] = []
    iniciais = problema.grupos_iniciais[indice_cadeia % len(problema.grupos_iniciais)]
    for regiao, grupos in zip(problema.regioes, iniciais):
        particoes.append(_criar_particao(regiao, grupos)[0] if regiao.k > 1 else None)

    aceitos = propostos = 0

    def avancar(fator: float) -> None:
        nonlocal aceitos, propostos
        for indice, regiao in enumerate(problema.regioes):
            if particoes[indice] is None:
                continue
            for _ in range(max(1, round(regiao.k * fator))):
                particoes[indice], aceito = _dar_passo(particoes[indice], regiao, problema, rng)
                aceitos += aceito
                propostos += 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BipartitionWarning)
        avancar(problema.n_burn_in)
        linhas = []
        contagem_vencedores = np.zeros(problema.votos.n_candidatos, dtype="int64")
        for _ in range(n_amostras):
            avancar(problema.passos_por_amostra)
            por_ag, vencedores = apurar_distritos(
                _vetor_de_distritos(particoes, problema), problema.votos, problema.n_distritos
            )
            linhas.append(por_ag)
            contagem_vencedores[vencedores] += 1
    return {
        "distritos": np.vstack(linhas),
        "vencedores": contagem_vencedores,
        "aceitos": aceitos,
        "propostos": propostos,
    }


def preparar_votos(
    votos_secao: pd.DataFrame,
    candidatos: pd.DataFrame,
    unidade_da_secao: pd.Series,
    secoes: pd.DataFrame,
    indice_unidade: dict[str, int],
    agremiacoes: list[str],
    chave_secao: list[str],
) -> tuple[Votos, list[str]]:
    """Agrega os votos por unidade e calcula a posição de cada candidato no desempate."""
    mapa_secao = secoes[chave_secao].assign(unidade=unidade_da_secao.map(indice_unidade).values)
    if mapa_secao["unidade"].isna().any():
        raise ValueError("Há seção sem unidade no conjunto de mapas.")
    df = votos_secao.merge(mapa_secao, on=chave_secao, how="inner", validate="many_to_one")
    if len(df) != len(votos_secao):
        raise ValueError("Há voto em seção sem unidade.")
    df = df.groupby(["unidade", "SQ_CANDIDATO", "agremiacao"], as_index=False)["votos"].sum()

    dados = candidatos[candidatos["SQ_CANDIDATO"].isin(df["SQ_CANDIDATO"])].copy()
    dados["_sq"] = dados["SQ_CANDIDATO"].astype("int64")
    dados = dados.sort_values(["DT_NASCIMENTO", "_sq"]).reset_index(drop=True)
    posicao = {sq: i for i, sq in enumerate(dados["SQ_CANDIDATO"])}
    indice_ag = {ag: i for i, ag in enumerate(agremiacoes)}
    return (
        Votos(
            unidades=df["unidade"].to_numpy(dtype="int64"),
            candidatos=df["SQ_CANDIDATO"].map(posicao).to_numpy(dtype="int64"),
            votos=df["votos"].to_numpy(dtype="float64"),
            n_candidatos=len(dados),
            ranking=np.arange(len(dados)),
            agremiacao=dados["agremiacao"].map(indice_ag).to_numpy(dtype="int64"),
            n_agremiacoes=len(agremiacoes),
        ),
        list(dados["SQ_CANDIDATO"]),
    )


def gerar_conjunto(
    secoes: pd.DataFrame,
    setores,
    votos_secao: pd.DataFrame,
    candidatos: pd.DataFrame,
    agremiacoes: list[str],
    chave_secao: list[str],
    n_distritos: int,
    config: dict,
) -> dict:
    """Gera as amostras de um mapa com `n_distritos`; devolve matrizes por amostra."""
    global _PROBLEMA

    secoes = secoes.reset_index(drop=True)
    alvo = distritos.calcular_alvo(int(secoes["QT_APTOS"].sum()), n_distritos)
    setores_metricos = distritos.para_metrico(setores).sort_values("CD_SETOR")
    tolerancia = config["tolerancia_eleitorado"]
    regioes, unidade_da_secao, _, _ = distritos._montar_regioes(
        secoes,
        setores_metricos,
        n_distritos,
        alvo,
        tolerancia,
        config["tolerancia_populacao_municipio"],
    )
    limites = (alvo * (1 - tolerancia), alvo * (1 + tolerancia))
    distritos._REGIOES = regioes
    distritos._LIMITES = limites
    distritos._MAX_TENTATIVAS = config["max_tentativas_corte"]
    distritos._MEMORIA_TOTAL_GB = config.get("memoria_maxima_gb_total")
    distritos.ajustar_pesos_das_regioes(
        regioes, list(range(distritos.N_SEMENTES_TESTE)), config["processos"]
    )

    # Pontos de partida: os primeiros mapas válidos, na mesma ordem de sementes do mapa de base.
    # Partidas diferentes permitem comparar as cadeias; se faltarem mapas, elas se repetem.
    inicio = config["semente_aleatoria"]
    sementes = list(range(inicio, inicio + config["n_sementes"]))
    n_cadeias = config["ensemble_cadeias"]
    iniciais = []
    with distritos.criar_piscina(config["processos"]) as piscina:
        for avaliacao in piscina.imap(distritos._avaliar_semente, sementes, chunksize=1):
            if avaliacao and avaliacao["media"] is not None:
                iniciais.append(avaliacao)
                if len(iniciais) == n_cadeias:
                    break
    if not iniciais:
        raise ValueError(f"Nenhum mapa inicial válido para {n_distritos} distritos.")

    primeira_unidade, primeiro_distrito, indice_unidade = [], [], {}
    unidade_atual = 0
    distrito_atual = 0
    for regiao in regioes:
        primeira_unidade.append(unidade_atual)
        primeiro_distrito.append(distrito_atual)
        for posicao, unidade in enumerate(regiao.unidades["unidade"]):
            indice_unidade[unidade] = unidade_atual + posicao
        unidade_atual += len(regiao.unidades)
        distrito_atual += regiao.k

    votos, candidatos_ordem = preparar_votos(
        votos_secao, candidatos, unidade_da_secao, secoes, indice_unidade, agremiacoes, chave_secao
    )
    _PROBLEMA = Problema(
        regioes=regioes,
        grupos_iniciais=[inicial["grupos"] for inicial in iniciais],
        primeira_unidade=primeira_unidade,
        primeiro_distrito=primeiro_distrito,
        votos=votos,
        limites=limites,
        n_distritos=n_distritos,
        max_tentativas=config["max_tentativas_corte"],
        n_burn_in=config["ensemble_burn_in"],
        passos_por_amostra=config["ensemble_passos_por_amostra"],
        semente_base=config["semente_aleatoria"] * 1000 + n_distritos,
    )

    por_cadeia = config["ensemble_amostras"] // n_cadeias
    trabalhos = [(indice, por_cadeia) for indice in range(n_cadeias)]
    with distritos.criar_piscina(min(config["processos"], n_cadeias)) as piscina:
        cadeias = piscina.map(_rodar_cadeia, trabalhos, chunksize=1)

    return {
        "distritos": np.vstack([cadeia["distritos"] for cadeia in cadeias]),
        "vencedores": sum(cadeia["vencedores"] for cadeia in cadeias),
        "candidatos": candidatos_ordem,
        "aceitacao": sum(c["aceitos"] for c in cadeias) / max(1, sum(c["propostos"] for c in cadeias)),
        "n_partidas": len(iniciais),
        "n_cadeias": n_cadeias,
        "regioes": [(regiao.nome, regiao.k, len(regiao.unidades)) for regiao in regioes],
    }


def calcular_gelman_rubin(serie: np.ndarray, n_cadeias: int) -> float:
    """Razão entre a variância total e a de dentro das cadeias; perto de 1 indica mistura."""
    cadeias = serie.reshape(n_cadeias, -1).astype("float64")
    n = cadeias.shape[1]
    dentro = cadeias.var(axis=1, ddof=1).mean()
    entre = n * cadeias.mean(axis=1).var(ddof=1)
    if dentro == 0:
        return 1.0 if entre == 0 else float("inf")
    return float(np.sqrt(((n - 1) / n * dentro + entre / n) / dentro))


def _cadeiras_e(cenario: dict, votos: pd.Series, n_cadeiras: int) -> pd.Series:
    return alocacao.alocar_proporcional(votos, n_cadeiras, cenario["metodo"], cenario["barreira"])


def _estatisticas(valores: np.ndarray, prefixo: str) -> dict[str, np.ndarray]:
    """Mínimo, quantis e máximo por coluna; os quantis são valores observados, sempre inteiros."""
    linhas = {f"{prefixo}_min": valores.min(axis=0)}
    for nome, quantil in QUANTIS.items():
        linhas[f"{prefixo}_{nome}"] = np.percentile(
            valores, quantil, axis=0, method="inverted_cdf"
        ).astype("int64")
    linhas[f"{prefixo}_max"] = valores.max(axis=0)
    return linhas


def resumir_cenarios(
    distritos_por_mapa: dict[int, np.ndarray],
    agremiacoes: list[str],
    votos: pd.Series,
    cenarios: list[dict],
    n_cadeiras: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Distribuição por agremiação e do tamanho da bancada do estado, em cada cenário PEC."""
    por_ag, por_estado = [], []
    for cenario in cenarios:
        if cenario["tipo"] != "pec":
            continue
        n_distritos = alocacao.calcular_n_distritos(
            n_cadeiras, Fraction(cenario["proporcao_distrital"])
        )
        amostras = distritos_por_mapa[n_distritos]
        direito = _cadeiras_e(cenario, votos, n_cadeiras).reindex(agremiacoes).to_numpy()
        totais = np.maximum(direito, amostras)
        excedentes = np.clip(amostras - direito, 0, None)
        por_ag.append(
            pd.DataFrame(
                {
                    "cenario": cenario["id"],
                    "ag": agremiacoes,
                    "cadeiras_E": direito,
                    **_estatisticas(amostras, "distritos"),
                    **_estatisticas(totais, "total"),
                }
            )
        )
        tamanhos = totais.sum(axis=1)
        por_estado.append(
            pd.DataFrame(
                {
                    "cenario": [cenario["id"]],
                    "n_amostras": [len(amostras)],
                    **{
                        chave: valor.ravel()
                        for chave, valor in _estatisticas(tamanhos[:, np.newaxis], "total").items()
                    },
                    "excedente_medio": [round(float(excedentes.sum(axis=1).mean()), 2)],
                    "prob_acima_de_70": [float((tamanhos > LIMITE_CONSTITUCIONAL).mean())],
                }
            )
        )
    return pd.concat(por_ag, ignore_index=True), pd.concat(por_estado, ignore_index=True)


def executar_uf(params: dict) -> None:
    """Gera os conjuntos de mapas de uma UF, grava as tabelas e desenha a figura."""
    pasta = carga.pasta_processados(params)
    saida = carga.pasta_saida(params)
    saida.mkdir(parents=True, exist_ok=True)
    config = params["distritos"]

    setores = carga.ler_setores(params)
    secoes = distritos.preparar_secoes(
        pd.read_parquet(pasta / "secoes.parquet"),
        setores,
        carga.ler_apelidos_municipios(params),
        carga.ler_locais_complementares(params),
    )
    candidatos = pd.read_parquet(pasta / "candidatos.parquet")
    votos_secao = pd.read_parquet(pasta / "votos_secao_candidato.parquet")
    votos = pd.read_parquet(pasta / "votos_ag.parquet").set_index("agremiacao")["votos"]
    votos = votos.sort_values(ascending=False, kind="stable")
    agremiacoes = list(votos.index)

    # O mesmo número de distritos é usado por vários cenários: basta gerar um conjunto.
    mapas = sorted(
        {
            alocacao.calcular_n_distritos(
                params["n_cadeiras"], Fraction(cenario["proporcao_distrital"])
            )
            for cenario in params["cenarios"]
            if cenario["tipo"] == "pec"
        }
    )
    distritos_por_mapa, amostras, diagnosticos, frequencias = {}, [], [], []
    nomes = candidatos.set_index("SQ_CANDIDATO")
    for n_distritos in mapas:
        conjunto = gerar_conjunto(
            secoes, setores, votos_secao, candidatos, agremiacoes, carga.CHAVE_SECAO, n_distritos, config
        )
        matriz = conjunto["distritos"]
        distritos_por_mapa[n_distritos] = matriz
        if (matriz.sum(axis=1) != n_distritos).any():
            raise ValueError("Amostra com número errado de distritos.")
        amostras.append(
            pd.DataFrame(matriz, columns=agremiacoes)
            .assign(mapa=n_distritos, amostra=range(len(matriz)))
            .melt(id_vars=["mapa", "amostra"], var_name="ag", value_name="distritos")
            .query("distritos > 0")
        )
        variaveis = matriz[:, matriz.var(axis=0) > 0]
        estatisticas = [
            calcular_gelman_rubin(variaveis[:, j], conjunto["n_cadeias"])
            for j in range(variaveis.shape[1])
        ]
        diagnosticos.append(
            {
                "mapa": n_distritos,
                "n_amostras": len(matriz),
                "n_cadeias": conjunto["n_cadeias"],
                "n_partidas": conjunto["n_partidas"],
                "aceitacao": round(conjunto["aceitacao"], 3),
                "composicoes_distintas": len({linha.tobytes() for linha in matriz}),
                "maior_gelman_rubin": round(max(estatisticas, default=1.0), 3),
            }
        )
        vitorias = conjunto["vencedores"]
        for posicao in np.flatnonzero(vitorias):
            sq = conjunto["candidatos"][posicao]
            frequencias.append(
                {
                    "mapa": n_distritos,
                    "SQ_CANDIDATO": sq,
                    "candidato": nomes.at[sq, "NM_URNA_CANDIDATO"],
                    "agremiacao": nomes.at[sq, "agremiacao"],
                    "frequencia": round(int(vitorias[posicao]) / len(matriz), 4),
                }
            )
        logger.info("Conjunto de %d distritos: %s", n_distritos, diagnosticos[-1])

    pd.concat(amostras, ignore_index=True).to_parquet(saida / "ensemble_amostras.parquet", index=False)
    pd.DataFrame(diagnosticos).to_csv(saida / "ensemble_diagnostico.csv", index=False)
    pd.DataFrame(frequencias).sort_values(
        ["mapa", "frequencia", "SQ_CANDIDATO"], ascending=[True, False, True]
    ).to_csv(saida / "ensemble_vencedores.csv", index=False)

    resumo, estado = resumir_cenarios(
        distritos_por_mapa, agremiacoes, votos, params["cenarios"], params["n_cadeiras"]
    )
    base = pd.read_csv(saida / "resultados_cenarios.csv")
    resumo = resumo.merge(
        base[["cenario", "ag", "distritos_D", "total"]].rename(
            columns={"distritos_D": "distritos_mapa_base", "total": "total_mapa_base"}
        ),
        on=["cenario", "ag"],
        how="left",
    )
    tamanho_base = base.groupby("cenario")["total"].sum().rename("total_mapa_base")
    estado = estado.merge(tamanho_base, on="cenario", how="left")
    resumo.to_csv(saida / "ensemble_resumo.csv", index=False)
    estado.to_csv(saida / "ensemble_estado.csv", index=False)

    nome_estado = carga.UFS[params["uf"]]["nome"]
    figuras.desenhar_figura_ensemble(
        resumo,
        estado,
        params["cenarios"],
        carga.ler_cores(),
        nome_estado,
        figuras.NOTA_ENSEMBLE.format(n_amostras=len(matriz), extra=""),
        saida / f"{distritos.normalizar_nome(nome_estado).lower().replace(' ', '_')}_"
        f"{params['ano']}_ensemble",
    )


def somar_brasil(
    amostras_por_uf: dict[str, pd.DataFrame],
    votos_por_uf: dict[str, pd.Series],
    n_cadeiras_por_uf: dict[str, int],
    cenarios: list[dict],
    semente: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Distribuição nacional, sorteando uma amostra de cada estado por amostra nacional.

    Os mapas de um estado não dependem dos de outro; a permutação evita que a ordem das cadeias
    crie correlação artificial entre estados.
    """
    agremiacoes = sorted({ag for votos in votos_por_uf.values() for ag in votos.index})
    indice = {ag: j for j, ag in enumerate(agremiacoes)}
    n_amostras = min(int(tabela["amostra"].max()) + 1 for tabela in amostras_por_uf.values())
    rng = np.random.default_rng(semente)
    por_ag, por_pais = [], []
    permutacoes = {uf: rng.permutation(n_amostras) for uf in sorted(amostras_por_uf)}
    for cenario in cenarios:
        if cenario["tipo"] != "pec":
            continue
        totais = np.zeros((n_amostras, len(agremiacoes)), dtype="int64")
        direitos = np.zeros(len(agremiacoes), dtype="int64")
        amostras_d = np.zeros((n_amostras, len(agremiacoes)), dtype="int64")
        for uf, tabela in amostras_por_uf.items():
            votos = votos_por_uf[uf]
            n_cadeiras = n_cadeiras_por_uf[uf]
            n_distritos = alocacao.calcular_n_distritos(
                n_cadeiras, Fraction(cenario["proporcao_distrital"])
            )
            direito = np.zeros(len(agremiacoes), dtype="int64")
            for ag, valor in _cadeiras_e(cenario, votos, n_cadeiras).items():
                direito[indice[ag]] = valor
            matriz = np.zeros((n_amostras, len(agremiacoes)), dtype="int64")
            parte = tabela[(tabela["mapa"] == n_distritos) & (tabela["amostra"] < n_amostras)]
            matriz[parte["amostra"].to_numpy(), parte["ag"].map(indice).to_numpy()] = parte["distritos"]
            matriz = matriz[permutacoes[uf]]
            totais += np.maximum(direito, matriz)
            amostras_d += matriz
            direitos += direito
        por_ag.append(
            pd.DataFrame(
                {
                    "cenario": cenario["id"],
                    "ag": agremiacoes,
                    "cadeiras_E": direitos,
                    **_estatisticas(amostras_d, "distritos"),
                    **_estatisticas(totais, "total"),
                }
            )
        )
        tamanhos = totais.sum(axis=1)[:, np.newaxis]
        por_pais.append(
            pd.DataFrame(
                {
                    "cenario": [cenario["id"]],
                    "n_amostras": [n_amostras],
                    **{chave: valor.ravel() for chave, valor in _estatisticas(tamanhos, "total").items()},
                    "excedente_medio": [round(float(tamanhos.mean()) - int(direitos.sum()), 2)],
                }
            )
        )
    return pd.concat(por_ag, ignore_index=True), pd.concat(por_pais, ignore_index=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argumentos = argparse.ArgumentParser(description=__doc__)
    argumentos.add_argument("--uf", required=True)
    argumentos.add_argument("--saida", type=Path, help="Pasta de saída em vez da de params.yaml.")
    argumentos.add_argument("--amostras", type=int, help="Substitui ensemble_amostras.")
    opcoes = argumentos.parse_args()
    parametros = carga.ler_params(opcoes.uf)
    if opcoes.saida:
        parametros["caminhos"]["outputs"] = str(opcoes.saida)
    if opcoes.amostras:
        parametros["distritos"]["ensemble_amostras"] = opcoes.amostras
    executar_uf(parametros)
