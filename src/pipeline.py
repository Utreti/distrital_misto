"""Orquestra as etapas da seção 4."""

from __future__ import annotations

import argparse
import logging
import time
from fractions import Fraction

import pandas as pd

import alocacao
import carga
import distritos
import figuras
import vencedores

logger = logging.getLogger(__name__)

LIMITE_CONSTITUCIONAL = 70
COLUNAS_RESULTADO = ["cadeiras_E", "distritos_D", "lista", "excedente", "total"]


def validar_2022(votos_ag: pd.DataFrame, candidatos: pd.DataFrame, n_cadeiras: int) -> pd.Series:
    """Etapa 3: compara o motor com o resultado oficial; divergência gera erro."""
    votos = votos_ag.set_index("agremiacao")["votos"]
    calculado = alocacao.alocar_proporcional(votos, n_cadeiras, "quociente_atual", barreira=True)
    calculado = calculado[calculado > 0].rename("cadeiras")
    oficial = carga.contar_eleitos_oficiais(candidatos)
    if not calculado.equals(oficial):
        raise ValueError(f"Motor difere do resultado oficial.\n{calculado}\n{oficial}")
    return calculado


def comparar_com_oficial(
    votos_ag: pd.DataFrame,
    votos_candidato: pd.DataFrame,
    candidatos: pd.DataFrame,
    n_cadeiras: int,
    coligacoes: pd.Series | None = None,
) -> pd.DataFrame:
    """Compara com o oficial a regra real e o motor dos cenários PEC (listas fechadas).

    Com `coligacoes` (partido e sua coligação), a regra real é a anterior ao fim das coligações.
    """
    votos = votos_ag.set_index("agremiacao")["votos"]
    regra_real = (
        alocacao.alocar_lista_aberta(votos, votos_candidato, n_cadeiras)
        if coligacoes is None
        else alocacao.alocar_com_coligacoes(votos, votos_candidato, coligacoes, n_cadeiras)
    )
    tabela = pd.DataFrame(
        {
            "lista_aberta": regra_real,
            "motor_pec": alocacao.alocar_proporcional(
                votos, n_cadeiras, "quociente_atual", barreira=True
            ),
            "oficial": carga.contar_eleitos_oficiais(candidatos),
        }
    ).fillna(0)
    tabela = tabela.astype(int).rename_axis("ag").reset_index()
    tabela["diferenca"] = tabela["lista_aberta"] - tabela["oficial"]
    return tabela[tabela[["lista_aberta", "motor_pec", "oficial"]].sum(axis=1) > 0]


def comparar_votos_secao(dados: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Soma das seções contra os totais oficiais, por agremiação e por candidato (3.2)."""
    secao_ag = pd.concat([dados["votos_secao_candidato"], dados["votos_secao_legenda"]])
    tabela_ag = pd.DataFrame(
        {
            "secao": secao_ag.groupby("agremiacao")["votos"].sum(),
            "oficial": dados["votos_ag"].set_index("agremiacao")["votos"]
            - dados["votos_ag"].set_index("agremiacao")["votos_convertidos"],
        }
    ).fillna(0)
    secao_cand = dados["votos_secao_candidato"].groupby("SQ_CANDIDATO")["votos"].sum()
    oficial_cand = dados["votos_candidato"].set_index("SQ_CANDIDATO")["votos"]
    oficial_cand = oficial_cand[oficial_cand > 0]
    tabela_cand = pd.DataFrame({"secao": secao_cand, "oficial": oficial_cand}).fillna(0)
    linhas = [
        tabela.astype(int).rename_axis("chave").reset_index().assign(nivel=nivel)
        for nivel, tabela in (("agremiacao", tabela_ag), ("candidato", tabela_cand))
    ]
    resultado = pd.concat(linhas, ignore_index=True)
    resultado["diferenca"] = resultado["secao"] - resultado["oficial"]
    return resultado[["nivel", "chave", "secao", "oficial", "diferenca"]]


def listar_mapas(params: dict) -> list[int]:
    """Número de distritos de cada mapa exigido pelos cenários (2.4)."""
    return sorted(
        {
            alocacao.calcular_n_distritos(
                params["n_cadeiras"], Fraction(cenario["proporcao_distrital"])
            )
            for cenario in params["cenarios"]
            if cenario["tipo"] == "pec"
        }
    )


def calcular_cenario(
    cenario: dict,
    votos: pd.Series,
    oficial: pd.Series,
    vencedores_por_mapa: dict[int, pd.DataFrame],
    votos_candidato: pd.DataFrame,
    n_cadeiras: int,
) -> pd.DataFrame:
    """Cadeiras por agremiação em um cenário, com todas as agremiações na tabela."""
    agremiacoes = votos.index
    if cenario["tipo"] == "pec":
        n_distritos = alocacao.calcular_n_distritos(
            n_cadeiras, Fraction(cenario["proporcao_distrital"])
        )
        cadeiras_e = alocacao.alocar_proporcional(
            votos, n_cadeiras, cenario["metodo"], cenario["barreira"]
        )
        distritos_d = vencedores_por_mapa[n_distritos].groupby("agremiacao")["distrito"].count()
        return alocacao.combinar_pec(cadeiras_e, distritos_d).reindex(agremiacoes)

    if cenario["tipo"] == "real":
        total = oficial.reindex(agremiacoes, fill_value=0).astype(int)
        return pd.DataFrame(
            {"cadeiras_E": total, "distritos_D": 0, "lista": total, "excedente": 0, "total": total}
        ).astype("Int64")

    if cenario["tipo"] == "distritao":
        eleitos = vencedores.apurar_distritao(votos_candidato, n_cadeiras)
        total = eleitos.groupby("agremiacao")["SQ_CANDIDATO"].count()
        total = total.reindex(agremiacoes, fill_value=0).astype(int)
        # No distritão não há direito proporcional, distrito nem lista: as colunas ficam vazias.
        return pd.DataFrame(
            {
                "cadeiras_E": pd.NA,
                "distritos_D": pd.NA,
                "lista": pd.NA,
                "excedente": 0,
                "total": total,
            },
            index=agremiacoes,
        ).astype("Int64")

    raise ValueError(f"Tipo de cenário desconhecido: {cenario['tipo']}")


def montar_resultados(
    params: dict,
    votos: pd.Series,
    oficial: pd.Series,
    vencedores_por_mapa: dict[int, pd.DataFrame],
    votos_candidato: pd.DataFrame,
) -> pd.DataFrame:
    """Tabela longa com todos os cenários; variação medida contra o resultado real."""
    # Agremiações na ordem fixa da figura: votos da eleição, do maior para o menor.
    votos = votos.sort_values(ascending=False, kind="stable")
    tabelas = []
    for cenario in params["cenarios"]:
        tabela = calcular_cenario(
            cenario, votos, oficial, vencedores_por_mapa, votos_candidato, params["n_cadeiras"]
        )
        tabelas.append(
            tabela[COLUNAS_RESULTADO]
            .astype("Int64")
            .rename_axis("ag")
            .reset_index()
            .assign(cenario=cenario["id"])
        )
    resultados = pd.concat(tabelas, ignore_index=True)
    real = resultados[resultados["cenario"] == params["cenarios"][0]["id"]].set_index("ag")["total"]
    resultados["variacao_vs_real"] = resultados["total"] - resultados["ag"].map(real)
    return resultados[["cenario", "ag", *COLUNAS_RESULTADO, "variacao_vs_real"]]


def montar_excedentes(resultados: pd.DataFrame, cenarios: list[dict], uf: str) -> pd.DataFrame:
    """Excedentes por cenário e agremiação; cenário sem excedente ocupa uma linha sem agremiação."""
    linhas = []
    for cenario in cenarios:
        tabela = resultados[resultados["cenario"] == cenario["id"]]
        total_estado = int(tabela["total"].sum())
        comuns = {
            "cenario": cenario["id"],
            "uf": uf,
            "total_estado": total_estado,
            "acima_de_70": total_estado > LIMITE_CONSTITUCIONAL,
        }
        com_excedente = tabela[tabela["excedente"] > 0]
        if com_excedente.empty:
            linhas.append({**comuns, "ag": "", "excedente": 0})
        for linha in com_excedente.itertuples():
            linhas.append({**comuns, "ag": linha.ag, "excedente": int(linha.excedente)})
    return pd.DataFrame(linhas)[
        ["cenario", "uf", "ag", "excedente", "total_estado", "acima_de_70"]
    ]


def montar_cenarios_metodos(params: dict) -> tuple[list[dict], list[dict]]:
    """Cenário real seguido de cada regra PEC em cada método; devolve também as linhas da grade."""
    regras = [cenario for cenario in params["cenarios"] if cenario["tipo"] == "pec"]
    linhas = [
        {"id": regra["id"], "rotulo": regra["rotulo"].replace(" · regra atual", "")}
        for regra in regras
    ]
    cenarios = [params["cenarios"][0]]
    for regra in regras:
        for metodo in params["metodos_comparados"]:
            cenarios.append(
                {
                    **regra,
                    "id": f"{regra['id']}__{metodo['id']}",
                    "metodo": metodo["id"],
                    "regra": regra["id"],
                }
            )
    return cenarios, linhas


def marcar_metodos(resultados: pd.DataFrame) -> pd.DataFrame:
    """Separa regra e método e mede cada método contra a regra atual, no mesmo mapa."""
    tabela = resultados.copy()
    partes = tabela["cenario"].str.split("__", expand=True).reindex(columns=[0, 1])
    tabela["regra"], tabela["metodo"] = partes[0], partes[1]
    atual = tabela[tabela["metodo"] == "quociente_atual"].set_index(["regra", "ag"])["total"]
    chaves = pd.MultiIndex.from_frame(tabela[["regra", "ag"]])
    tabela["diferenca_vs_regra_atual"] = tabela["total"].to_numpy() - atual.reindex(chaves).to_numpy()
    return tabela


def etapa_metodos(
    params: dict,
    votos: pd.Series,
    oficial: pd.Series,
    vencedores_por_mapa: dict[int, pd.DataFrame],
    votos_candidato: pd.DataFrame,
    caminho_base,
) -> None:
    """Fase 2: as regras PEC em cada método de alocação, nos mesmos mapas de base."""
    cenarios, linhas = montar_cenarios_metodos(params)
    resultados = montar_resultados(
        {**params, "cenarios": cenarios}, votos, oficial, vencedores_por_mapa, votos_candidato
    )
    marcar_metodos(resultados).to_csv(
        carga.pasta_saida(params) / "resultados_metodos.csv", index=False
    )
    figuras.desenhar_figura_metodos(
        resultados,
        linhas,
        params["metodos_comparados"],
        cenarios[0]["id"],
        carga.ler_cores(),
        carga.UFS[params["uf"]]["nome"],
        caminho_base,
    )


def etapa_carga_e_validacao(params: dict) -> None:
    """Etapas 1 a 3; divergências ficam registradas em validacao_*.csv e no log."""
    pasta = carga.pasta_processados(params)
    pasta.mkdir(parents=True, exist_ok=True)
    saida = carga.pasta_saida(params)
    saida.mkdir(parents=True, exist_ok=True)

    dados = carga.carregar_dados(params)
    for nome, df in dados.items():
        df.to_parquet(pasta / f"{nome}.parquet", index=False)

    coligacoes = None
    if params["ano"] in params["anos_com_coligacoes"]:
        coligacoes = dados["votos_partido"].set_index("agremiacao")["SQ_COLIGACAO"]
    comparacao = comparar_com_oficial(
        dados["votos_ag"],
        dados["votos_candidato"],
        dados["candidatos"],
        params["n_cadeiras"],
        coligacoes,
    )
    comparacao.to_csv(saida / "validacao_oficial.csv", index=False)
    if (comparacao["diferenca"] != 0).any():
        logger.warning("Motor difere do resultado oficial:\n%s", comparacao.to_string(index=False))
    else:
        logger.info("Validação contra o oficial aprovada: %d cadeiras", params["n_cadeiras"])

    votos = comparar_votos_secao(dados)
    votos.to_csv(saida / "validacao_votos_secao.csv", index=False)
    divergentes = votos[votos["diferenca"] != 0]
    logger.log(
        logging.WARNING if len(divergentes) else logging.INFO,
        "Seções contra totais oficiais: %d divergências em %d linhas",
        len(divergentes),
        len(votos),
    )


def etapa_distritos(params: dict) -> None:
    """Etapa 4: mapas de base, a partir das seções já processadas."""
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
    fora = distritos.listar_locais_fora_do_municipio(secoes, setores)
    logger.info("Locais fora do município do TSE: %d", len(fora))
    inicio = config["semente_aleatoria"]
    sementes = list(range(inicio, inicio + config["n_sementes"]))

    secoes_por_mapa, resumos, selecoes = [], [], []
    for n_distritos in listar_mapas(params):
        comeco = time.perf_counter()
        mapa = distritos.selecionar_mapa(
            secoes,
            setores,
            n_distritos,
            sementes,
            config["tolerancia_eleitorado"],
            config["tolerancia_populacao_municipio"],
            config["max_tentativas_corte"],
            config["processos"],
            config.get("parar_na_primeira", False),
            config.get("memoria_maxima_gb_total"),
        )
        logger.info(
            "Mapa de %d distritos: semente %d, %d válidos, %d falhas, Polsby-Popper médio %.4f, "
            "%.0f s",
            n_distritos,
            mapa.semente,
            mapa.n_validos,
            mapa.n_falhas,
            mapa.polsby_popper_medio,
            time.perf_counter() - comeco,
        )
        secoes_por_mapa.append(mapa.secao_distrito.assign(mapa=n_distritos))
        resumos.append(distritos.resumir_mapa(mapa))
        selecoes.append(
            {
                "mapa": n_distritos,
                "semente": mapa.semente,
                "n_sementes": len(sementes),
                "n_validos": mapa.n_validos,
                "n_falhas": mapa.n_falhas,
                "polsby_popper_medio": round(mapa.polsby_popper_medio, 4),
                "municipios_nao_inteiros": " ".join(mapa.municipios_nao_inteiros),
            }
        )
        figuras.desenhar_mapa_distritos(
            mapa.geometrias,
            mapa.rotulos,
            secoes,
            mapa.distritos_zoom,
            saida / f"mapa_distritos_{n_distritos}.png",
        )

    colunas = ["mapa", "distrito", *carga.CHAVE_SECAO]
    pd.concat(secoes_por_mapa)[colunas].to_parquet(pasta / "secao_distrito.parquet", index=False)
    pd.concat(resumos).to_parquet(pasta / "distritos_base.parquet", index=False)
    pd.DataFrame(selecoes).to_csv(saida / "mapas_selecao.csv", index=False)
    fora.to_csv(saida / "locais_fora_do_municipio.csv", index=False)


def etapa_resultados(params: dict) -> None:
    """Etapas 5 a 8: vencedores, regra PEC, distritão, tabelas e figura."""
    pasta = carga.pasta_processados(params)
    saida = carga.pasta_saida(params)
    saida.mkdir(parents=True, exist_ok=True)

    candidatos = pd.read_parquet(pasta / "candidatos.parquet")
    votos_secao = pd.read_parquet(pasta / "votos_secao_candidato.parquet")
    votos_candidato = pd.read_parquet(pasta / "votos_candidato.parquet")
    votos_ag = pd.read_parquet(pasta / "votos_ag.parquet").set_index("agremiacao")["votos"]
    secao_distrito = pd.read_parquet(pasta / "secao_distrito.parquet")

    vencedores_por_mapa = {}
    for n_distritos in listar_mapas(params):
        votos_distrito = vencedores.agregar_votos_distrito(
            votos_secao, secao_distrito[secao_distrito["mapa"] == n_distritos], carga.CHAVE_SECAO
        )
        vencedores_por_mapa[n_distritos] = vencedores.apurar_vencedores(votos_distrito, candidatos)

    tabela_vencedores = pd.concat(
        [tabela.assign(mapa=mapa) for mapa, tabela in vencedores_por_mapa.items()]
    ).rename(
        columns={
            "NM_URNA_CANDIDATO": "vencedor",
            "agremiacao": "agremiacao_vencedor",
            "votos": "votos_vencedor",
        }
    )
    tabela_vencedores.to_parquet(pasta / "vencedores_distritos.parquet", index=False)
    resumo = pd.read_parquet(pasta / "distritos_base.parquet").merge(
        tabela_vencedores[
            ["mapa", "distrito", "vencedor", "agremiacao_vencedor", "votos_vencedor"]
        ],
        on=["mapa", "distrito"],
        how="left",
        validate="one_to_one",
    )
    resumo.to_csv(saida / "distritos_resumo.csv", index=False)

    oficial = carga.contar_eleitos_oficiais(candidatos)
    resultados = montar_resultados(
        params, votos_ag, oficial, vencedores_por_mapa, votos_candidato
    )
    resultados.to_csv(saida / "resultados_cenarios.csv", index=False)
    excedentes = montar_excedentes(resultados, params["cenarios"], params["uf"])
    excedentes.to_csv(saida / "excedentes.csv", index=False)
    if excedentes["acima_de_70"].any():
        logger.warning("Há cenário acima de %d cadeiras no estado.", LIMITE_CONSTITUCIONAL)

    nome_estado = carga.UFS[params["uf"]]["nome"]
    slug = f"{distritos.normalizar_nome(nome_estado).lower().replace(' ', '_')}_{params['ano']}"
    etapa_metodos(
        params, votos_ag, oficial, vencedores_por_mapa, votos_candidato, saida / f"{slug}_metodos"
    )
    selecao = pd.read_csv(saida / "mapas_selecao.csv", dtype=str, keep_default_na=False)
    codigos = sorted(set(" ".join(selecao["municipios_nao_inteiros"]).split()))
    setores = carga.ler_setores(params)
    nomes_nao_inteiros = sorted(setores.loc[setores["CD_MUN"].isin(codigos), "NM_MUN"].unique())
    figuras.desenhar_figura_cenarios(
        resultados,
        params["cenarios"],
        carga.ler_cores(),
        nome_estado,
        ", ".join(nomes_nao_inteiros[:-1]) + " e " + nomes_nao_inteiros[-1]
        if len(nomes_nao_inteiros) > 1
        else "".join(nomes_nao_inteiros) or "nenhum",
        saida / f"{slug}_fase1",
        params["uf"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argumentos = argparse.ArgumentParser(description=__doc__)
    argumentos.add_argument("--uf", help="UF a simular; o padrão está em params.yaml.")
    argumentos.add_argument("--ano", type=int, help="Eleição; o padrão está em params.yaml.")
    argumentos.add_argument(
        "--etapa",
        choices=["tudo", "distritos", "resultados"],
        default="tudo",
        help="Reexecuta só a etapa escolhida e as seguintes, usando os parquets já gerados.",
    )
    argumentos.add_argument("--sementes", type=int, help="Substitui n_sementes de params.yaml.")
    argumentos.add_argument("--saida", help="Pasta de saída em vez da definida em params.yaml.")
    argumentos.add_argument(
        "--um-mapa", action="store_true", help="Para na primeira semente que gerar mapa válido."
    )
    opcoes = argumentos.parse_args()
    parametros = carga.ler_params(opcoes.uf, opcoes.ano)
    figuras.ANO = parametros["ano"]
    if opcoes.sementes:
        parametros["distritos"]["n_sementes"] = opcoes.sementes
    if opcoes.saida:
        parametros["caminhos"]["outputs"] = opcoes.saida
    parametros["distritos"]["parar_na_primeira"] = opcoes.um_mapa
    if opcoes.etapa == "tudo":
        etapa_carga_e_validacao(parametros)
    if opcoes.etapa in ("tudo", "distritos"):
        etapa_distritos(parametros)
    etapa_resultados(parametros)
