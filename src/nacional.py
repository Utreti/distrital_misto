"""Roda várias UFs, relata problemas por estado e monta o resultado nacional."""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

import carga
import distritos
import ensemble
import figuras
import pipeline

logger = logging.getLogger(__name__)

TEMPO_MAXIMO_UF = 3 * 3600
TEMPO_MAXIMO_ENSEMBLE = 8 * 3600
SUFIXOS_FIGURAS = ("fase1", "metodos", "ensemble")


def executar_uf(uf: str, saida: Path, um_mapa: bool) -> dict:
    """Roda uma UF em processo próprio, para que a falha de um estado não pare os demais."""
    pasta = saida / uf
    pasta.mkdir(parents=True, exist_ok=True)
    comando = [sys.executable, str(carga.RAIZ / "src" / "pipeline.py"), "--uf", uf]
    comando += ["--saida", str(saida)]
    if um_mapa:
        comando.append("--um-mapa")
    comeco = time.perf_counter()
    try:
        processo = subprocess.run(
            comando, capture_output=True, text=True, timeout=TEMPO_MAXIMO_UF, check=False
        )
        log = processo.stdout + processo.stderr
        status = "ok" if processo.returncode == 0 else "falha"
    except subprocess.TimeoutExpired as erro:
        log = f"{erro.stdout or ''}{erro.stderr or ''}\nTempo máximo excedido"
        status = "tempo_excedido"
    (pasta / "execucao.log").write_text(str(log), encoding="utf-8")
    return resumir_log(uf, status, str(log), time.perf_counter() - comeco, pasta)


def executar_ensemble(uf: str, saida: Path) -> tuple[str, float, str]:
    """Gera o conjunto de mapas da UF em processo próprio; requer o mapa de base já pronto."""
    comando = [sys.executable, str(carga.RAIZ / "src" / "ensemble.py"), "--uf", uf]
    comando += ["--saida", str(saida)]
    comeco = time.perf_counter()
    try:
        processo = subprocess.run(
            comando, capture_output=True, text=True, timeout=TEMPO_MAXIMO_ENSEMBLE, check=False
        )
        log = processo.stdout + processo.stderr
        status = "ok" if processo.returncode == 0 else "falha"
    except subprocess.TimeoutExpired as erro:
        log = f"{erro.stdout or ''}{erro.stderr or ''}\nTempo máximo excedido"
        status = "tempo_excedido"
    (saida / uf / "ensemble.log").write_text(str(log), encoding="utf-8")
    erros = [linha for linha in str(log).splitlines() if re.match(r"^\w*(Error|Exception)\b", linha)]
    return status, time.perf_counter() - comeco, erros[-1][:300] if erros else ""


def resumir_log(uf: str, status: str, log: str, segundos: float, pasta: Path) -> dict:
    """Uma linha do relatório: o que aconteceu de diferente do caso simples."""
    linhas = log.splitlines()
    erros = [linha for linha in linhas if re.match(r"^\w*(Error|Exception)\b", linha)]

    def extrair(padrao: str) -> str:
        achados = {m.group(1) for linha in linhas if (m := re.search(padrao, linha))}
        return "; ".join(sorted(achados))

    relatorio = {
        "uf": uf,
        "status": status,
        "segundos": round(segundos),
        "erro": erros[-1][:300] if erros else "",
        "distrito_proprio": extrair(r"Distrito próprio obrigatório: (.+)"),
        "encurralados": extrair(r"(?:Encurralado|Isolado|Ajuste de distritos): (.+?) dividirá"),
        "vizinhos_divididos": extrair(r"Vizinho dividido: (.+?),"),
        "peso_eleitorado": extrair(r"Peso por eleitorado: (.+?) \("),
        "ligacoes": extrair(r"Contiguidade: (.+?) \("),
        "coordenadas": extrair(r"Coordenadas imputadas: (.+)"),
        "sem_data_nascimento": extrair(r"sem data de nascimento: (.+)"),
    }
    validacao = pasta / "validacao_2022.csv"
    if validacao.exists():
        tabela = pd.read_csv(validacao)
        relatorio["n_cadeiras"] = int(tabela["oficial"].sum())
        divergentes = tabela[tabela["diferenca"] != 0]
        relatorio["validacao_divergente"] = "; ".join(
            f"{linha.ag} {linha.diferenca:+d}" for linha in divergentes.itertuples()
        )
    votos = pasta / "validacao_votos_secao.csv"
    if votos.exists():
        tabela = pd.read_csv(votos)
        relatorio["votos_secao_divergentes"] = int((tabela["diferenca"] != 0).sum())
    selecao = pasta / "mapas_selecao.csv"
    if selecao.exists():
        tabela = pd.read_csv(selecao, dtype=str, keep_default_na=False)
        relatorio["mapas"] = " ".join(
            f"{l.mapa}:semente {l.semente}" for l in tabela.itertuples()
        )
    excedentes = pasta / "excedentes.csv"
    if excedentes.exists():
        tabela = pd.read_csv(excedentes)
        relatorio["maior_total"] = int(tabela["total_estado"].max())
        relatorio["acima_de_70"] = bool(tabela["acima_de_70"].any())
    return relatorio


def agregar_estados(pastas: dict[str, Path]) -> pd.DataFrame:
    """Soma os resultados das UFs por cenário e agremiação, mantendo a linha de cada UF."""
    tabelas = [
        pd.read_csv(pasta / "resultados_cenarios.csv").assign(uf=uf) for uf, pasta in pastas.items()
    ]
    return pd.concat(tabelas, ignore_index=True)


def somar_brasil(por_estado: pd.DataFrame, ordem: list[str]) -> pd.DataFrame:
    """Resultado nacional, com as agremiações na ordem de votos de 2022."""
    colunas = ["cadeiras_E", "distritos_D", "lista", "excedente", "total", "variacao_vs_2022"]
    soma = por_estado.groupby(["cenario", "ag"], as_index=False, sort=False)[colunas].sum()
    soma["_ordem"] = soma["ag"].map({ag: i for i, ag in enumerate(ordem)})
    cenarios = {c: i for i, c in enumerate(por_estado["cenario"].unique())}
    soma["_cenario"] = soma["cenario"].map(cenarios)
    soma = soma.sort_values(["_cenario", "_ordem"]).drop(columns=["_ordem", "_cenario"])
    return soma.reset_index(drop=True)


def ordenar_por_votos(params: dict, ufs: list[str]) -> list[str]:
    tabelas = [
        pd.read_parquet(carga.RAIZ / params["caminhos"]["processed"] / uf / "votos_ag.parquet")
        for uf in ufs
    ]
    votos = pd.concat(tabelas).groupby("agremiacao")["votos"].sum()
    return list(votos.sort_values(ascending=False, kind="stable").index)


def montar_brasil(params: dict, pastas: dict[str, Path], saida: Path) -> pd.DataFrame:
    """Grava as tabelas nacionais e a figura do país."""
    por_estado = agregar_estados(pastas)
    por_estado.to_csv(saida / "resultados_por_estado.csv", index=False)
    ordem = ordenar_por_votos(params, list(pastas))
    brasil = somar_brasil(por_estado, ordem)
    brasil.to_csv(saida / "resultados_brasil.csv", index=False)
    figuras.desenhar_figura_nacional(
        brasil, params["cenarios"], carga.ler_cores(), saida / f"brasil_{params['ano']}_fase1"
    )
    metodos = pd.concat(
        pd.read_csv(pasta / "resultados_metodos.csv").assign(uf=uf) for uf, pasta in pastas.items()
    )
    cenarios, linhas = pipeline.montar_cenarios_metodos(params)
    soma = somar_brasil(metodos.drop(columns=["regra", "metodo", "diferenca_vs_regra_atual"]), ordem)
    pipeline.marcar_metodos(soma).to_csv(saida / "resultados_metodos_brasil.csv", index=False)
    figuras.desenhar_figura_metodos(
        soma,
        linhas,
        params["metodos_comparados"],
        cenarios[0]["id"],
        carga.ler_cores(),
        "Brasil",
        saida / f"brasil_{params['ano']}_metodos",
    )
    return brasil


def montar_ensemble_brasil(params: dict, pastas: dict[str, Path], saida: Path) -> None:
    """Soma nacional do conjunto de mapas, com o resultado do mapa de base para comparação."""
    amostras, votos, cadeiras = {}, {}, {}
    for uf, pasta in pastas.items():
        amostras[uf] = pd.read_parquet(pasta / "ensemble_amostras.parquet")
        parametros_uf = carga.ler_params(uf)
        cadeiras[uf] = parametros_uf["n_cadeiras"]
        votos[uf] = pd.read_parquet(
            carga.pasta_processados(parametros_uf) / "votos_ag.parquet"
        ).set_index("agremiacao")["votos"]
    resumo, pais = ensemble.somar_brasil(
        amostras, votos, cadeiras, params["cenarios"], params["distritos"]["semente_aleatoria"]
    )
    base = pd.read_csv(saida / "resultados_brasil.csv")
    ordem = {ag: i for i, ag in enumerate(ordenar_por_votos(params, list(pastas)))}
    resumo = resumo.merge(
        base[["cenario", "ag", "distritos_D", "total"]].rename(
            columns={"distritos_D": "distritos_mapa_base", "total": "total_mapa_base"}
        ),
        on=["cenario", "ag"],
        how="left",
    )
    resumo = resumo.sort_values(["cenario", "ag"], key=lambda c: c.map(ordem) if c.name == "ag" else c)
    pais = pais.merge(
        base.groupby("cenario")["total"].sum().rename("total_mapa_base"), on="cenario", how="left"
    )
    resumo.to_csv(saida / "ensemble_resumo_brasil.csv", index=False)
    pais.to_csv(saida / "ensemble_brasil.csv", index=False)
    n_amostras = int(pais["n_amostras"].iloc[0])
    figuras.desenhar_figura_ensemble(
        resumo,
        pais,
        params["cenarios"],
        carga.ler_cores(),
        "Brasil",
        figuras.NOTA_ENSEMBLE.format(
            n_amostras=n_amostras,
            extra="A soma nacional sorteia, de forma independente, um mapa de cada estado.",
        ),
        saida / f"brasil_{params['ano']}_ensemble",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argumentos = argparse.ArgumentParser(description=__doc__)
    argumentos.add_argument("--saida", required=True, type=Path)
    argumentos.add_argument("--ufs", nargs="+", default=sorted(carga.UFS))
    argumentos.add_argument("--um-mapa", action="store_true")
    argumentos.add_argument("--ensemble", action="store_true", help="Roda também a Fase 3.")
    argumentos.add_argument(
        "--reaproveitar",
        nargs="*",
        default=[],
        metavar="UF=PASTA",
        help="UFs já prontas, somadas ao resultado nacional sem rodar de novo.",
    )
    opcoes = argumentos.parse_args()
    parametros = carga.ler_params()
    opcoes.saida.mkdir(parents=True, exist_ok=True)

    relatorios = []
    for uf in opcoes.ufs:
        relatorio = executar_uf(uf, opcoes.saida, opcoes.um_mapa)
        if opcoes.ensemble and relatorio["status"] == "ok":
            status, segundos, erro = executar_ensemble(uf, opcoes.saida)
            relatorio.update(ensemble=status, ensemble_segundos=round(segundos), ensemble_erro=erro)
        relatorios.append(relatorio)
        logger.info("%s: %s em %ss %s", uf, relatorio["status"], relatorio["segundos"], relatorio["erro"])
        # A cada estado, para que uma interrupção não perca o relatório.
        pd.DataFrame(relatorios).to_csv(opcoes.saida / "relatorio_estados.csv", index=False)

    pastas = {uf: opcoes.saida / uf for uf, rel in zip(opcoes.ufs, relatorios) if rel["status"] == "ok"}
    for item in opcoes.reaproveitar:
        uf, pasta = item.split("=")
        pastas[uf] = Path(pasta)
    faltantes = sorted(set(carga.UFS) - set(pastas))
    if faltantes:
        logger.warning("Resultado nacional incompleto; faltam: %s", faltantes)
    else:
        montar_brasil(parametros, dict(sorted(pastas.items())), opcoes.saida)
        if opcoes.ensemble:
            montar_ensemble_brasil(parametros, dict(sorted(pastas.items())), opcoes.saida)
    for uf, pasta in pastas.items():
        nome = distritos.normalizar_nome(carga.UFS[uf]["nome"]).lower().replace(" ", "_")
        for tipo in SUFIXOS_FIGURAS:
            for extensao in (".png", ".svg"):
                origem = pasta / f"{nome}_{parametros['ano']}_{tipo}{extensao}"
                if origem.exists():
                    shutil.copy(origem, opcoes.saida / origem.name)
