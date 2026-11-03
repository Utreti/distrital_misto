"""Figuras do projeto: bolinhas por cenário (seção 6) e mapa de diagnóstico (seção 5.3)."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CORES_DISTRITOS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]

COR_TEXTO = "#1F2328"
COR_TEXTO_SECUNDARIO = "#57606A"
COR_FUNDO = "#FFFFFF"
COR_ANEL = "#000000"
TAMANHO_BOLINHA = 300
TAMANHO_ANEL = 560
MAX_BOLINHAS_POR_LINHA = 30
# Medidas verticais em polegadas.
ALTURA_CENARIO = 0.853
ALTURA_LINHA = 0.27
FOLGA_CENARIO = 0.42
MARGEM_INFERIOR = 2.16
MARGEM_SUPERIOR = 0.72

NOTAS = (
    "Notas: contrafactual mecânico com os votos de 2022; não modela mudanças de comportamento de "
    "eleitores, candidatos ou partidos. Federações contam como um único partido. Vencedor do "
    "distrito: mais votos, no máximo um distrito por candidato, e o candidato mais velho vence "
    "empates. Empates na alocação proporcional vão para a agremiação com mais votos no total "
    "(hipótese de modelagem). Excedentes sem limite. Barreira = 80% do quociente eleitoral. "
    "Distritos desenhados por contiguidade e equilíbrio de ±20% do eleitorado, com municípios "
    "inteiros, exceto {nao_inteiros}. {ressalva}"
)


def _formatar_variacoes(tabela: pd.DataFrame) -> str:
    """Saldo por agremiação contra 2022, como "MDB +1 · PP −1"."""
    mudancas = tabela[tabela["variacao_vs_2022"] != 0]
    partes = [
        f"{linha.ag} {'+' if linha.variacao_vs_2022 > 0 else '−'}{abs(linha.variacao_vs_2022)}"
        for linha in mudancas.itertuples()
    ]
    return " · ".join(partes) if partes else "sem mudança"


def contar_linhas(n_bolinhas: int) -> int:
    """Linhas de bolinhas necessárias para um cenário."""
    return max(1, math.ceil(n_bolinhas / MAX_BOLINHAS_POR_LINHA))


def desenhar_figura_cenarios(
    resultados: pd.DataFrame,
    cenarios: list[dict],
    cores: dict[str, str],
    nome_estado: str,
    nao_inteiros: str,
    caminho_base: Path,
) -> None:
    """Um bloco por cenário e uma bolinha por cadeira; grava PNG (300 dpi) e SVG.

    Cada linha tem no máximo MAX_BOLINHAS_POR_LINHA bolinhas. A ordem de `resultados` define a
    ordem fixa das agremiações em todos os cenários.
    """
    presentes = list(resultados.loc[resultados["total"] > 0, "ag"].unique())
    sem_cor = [ag for ag in presentes if ag not in cores]
    if sem_cor:
        raise ValueError(f"Agremiações sem cor em cores_agremiacoes.csv: {sem_cor}")

    totais = resultados.groupby("cenario")["total"].sum()
    colunas = min(int(totais.max()), MAX_BOLINHAS_POR_LINHA)

    # As medidas verticais são em polegadas; em estados grandes a figura cresce para baixo.
    alturas = [
        max(ALTURA_CENARIO, contar_linhas(int(totais[cenario["id"]])) * ALTURA_LINHA + FOLGA_CENARIO)
        for cenario in cenarios
    ]
    altura_eixo = sum(alturas)
    altura_figura = MARGEM_INFERIOR + altura_eixo + MARGEM_SUPERIOR
    figura = plt.figure(figsize=(15, altura_figura), facecolor=COR_FUNDO)
    eixo = figura.add_axes(
        [0.235, MARGEM_INFERIOR / altura_figura, 0.755, altura_eixo / altura_figura]
    )
    eixo.set_axis_off()
    # O texto à direita ocupa espaço fixo; com muitas colunas as bolinhas encolhem para caber.
    espaco_texto = max(13, colunas * 0.7)
    eixo.set_xlim(-0.7, colunas + espaco_texto)
    eixo.set_ylim(altura_eixo, 0)
    passo = 15 * 0.755 * 72 / (colunas + espaco_texto + 0.7)
    escala = min(1.0, (0.85 * passo) ** 2 / TAMANHO_BOLINHA)
    caracteres = int(espaco_texto / (colunas + espaco_texto + 0.7) * 15 * 0.755 * 13.5)

    topo = 0.0
    for cenario, altura in zip(cenarios, alturas):
        centro = topo + altura / 2
        topo += altura
        tabela = resultados[resultados["cenario"] == cenario["id"]]
        total = int(tabela["total"].sum())
        n_linhas = contar_linhas(total)
        posicao = 0
        for linha in tabela[tabela["total"] > 0].itertuples():
            ganhos = max(0, int(linha.variacao_vs_2022))
            for cadeira in range(int(linha.total)):
                # Excedentes e cadeiras ganhas ficam no fim do grupo da agremiação.
                excedente = cadeira >= linha.total - linha.excedente
                ganho = cadeira >= linha.total - ganhos
                marcador = "s" if excedente else "o"
                x = posicao % MAX_BOLINHAS_POR_LINHA
                y = (
                    centro
                    + (posicao // MAX_BOLINHAS_POR_LINHA - (n_linhas - 1) / 2) * ALTURA_LINHA
                )
                if ganho:
                    eixo.scatter(
                        x,
                        y,
                        s=TAMANHO_ANEL * escala,
                        marker=marcador,
                        facecolor="none",
                        edgecolor=COR_ANEL,
                        linewidth=2.4 * max(escala, 0.4),
                        zorder=2,
                    )
                eixo.scatter(
                    x,
                    y,
                    s=TAMANHO_BOLINHA * escala,
                    marker=marcador,
                    color=cores[linha.ag],
                    edgecolor=COR_FUNDO,
                    linewidth=max(escala, 0.3),
                    zorder=3,
                )
                posicao += 1

        if cenario is cenarios[0]:
            composicao = tabela[tabela["total"] > 0]
            detalhe = " · ".join(f"{l.ag} {l.total}" for l in composicao.itertuples())
        else:
            detalhe = _formatar_variacoes(tabela)
        # O total sobe conforme o número de linhas do detalhe, para os dois não se sobreporem.
        texto = textwrap.fill(detalhe, width=caracteres)
        recuo = 0.111 * (texto.count("\n") + 1)
        eixo.text(
            colunas + 0.1,
            centro - recuo,
            f"{total} cadeiras",
            color=COR_TEXTO,
            fontsize=10.5,
            fontweight="bold",
            va="center",
        )
        eixo.text(
            colunas + 0.1,
            centro - recuo + 0.102,
            texto,
            color=COR_TEXTO_SECUNDARIO,
            fontsize=9.5,
            va="top",
        )
        eixo.text(
            -0.75, centro, cenario["rotulo"], color=COR_TEXTO, fontsize=10.5, va="center", ha="right"
        )

    figura.text(
        0.012,
        (altura_figura - 0.36) / altura_figura,
        f"Cadeiras de deputado federal: {nome_estado}, sob cada regra, com os votos de 2022",
        color=COR_TEXTO,
        fontsize=14,
        fontweight="bold",
    )

    n_cadeiras = int(resultados[resultados["cenario"] == cenarios[0]["id"]]["total"].sum())
    ressalva = (
        f"Com {n_cadeiras} cadeiras, os resultados são um piloto, não uma conclusão."
        if n_cadeiras <= 10
        else "Resultado de um único mapa distrital; outros mapas podem mudar os vencedores."
    )
    legenda = [
        Line2D([], [], marker="o", linestyle="", color=cores[ag], markersize=11, label=ag)
        for ag in presentes
    ]
    legenda += [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="none",
            markeredgecolor=COR_ANEL,
            markeredgewidth=2.2,
            markersize=13,
            label="Cadeira ganha em relação a 2022",
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            color=COR_TEXTO_SECUNDARIO,
            markersize=11,
            label="Cadeira excedente",
        ),
    ]
    figura.legend(
        handles=legenda,
        loc="lower left",
        bbox_to_anchor=(0.012, 1.08 / altura_figura),
        ncol=5,
        frameon=False,
        fontsize=10,
        labelcolor=COR_TEXTO,
        handletextpad=0.3,
        columnspacing=1.6,
    )
    figura.text(
        0.012,
        0.16 / altura_figura,
        textwrap.fill(NOTAS.format(nao_inteiros=nao_inteiros, ressalva=ressalva), width=175),
        color=COR_TEXTO_SECUNDARIO,
        fontsize=9,
        va="bottom",
    )

    figura.savefig(f"{caminho_base}.png", dpi=300, metadata={"Software": None})
    # A semente fixa e a data vazia mantêm o SVG idêntico entre execuções.
    with plt.rc_context({"svg.hashsalt": "pec-distrital"}):
        figura.savefig(f"{caminho_base}.svg", metadata={"Date": None, "Creator": None})
    plt.close(figura)


def desenhar_figura_nacional(
    resultados: pd.DataFrame, cenarios: list[dict], cores: dict[str, str], caminho_base: Path
) -> None:
    """Um bloco de bolinhas por cenário, com no máximo 30 por linha, em grade 3 x 2."""
    presentes = list(resultados.loc[resultados["total"] > 0, "ag"].unique())
    sem_cor = [ag for ag in presentes if ag not in cores]
    if sem_cor:
        raise ValueError(f"Agremiações sem cor em cores_agremiacoes.csv: {sem_cor}")

    colunas_grade = 3
    linhas_grade = math.ceil(len(cenarios) / colunas_grade)
    totais = resultados.groupby("cenario")["total"].sum()
    linhas_bloco = contar_linhas(int(totais.max()))
    passo = 0.2
    largura_bloco = MAX_BOLINHAS_POR_LINHA * passo + 0.5
    altura_bloco = linhas_bloco * passo + 1.5
    largura = largura_bloco * colunas_grade + 0.4
    rodape = 2.1
    altura = altura_bloco * linhas_grade + rodape + 0.7

    figura = plt.figure(figsize=(largura, altura), facecolor=COR_FUNDO)
    eixo = figura.add_axes([0, 0, 1, 1])
    eixo.set_axis_off()
    eixo.set_xlim(0, largura)
    eixo.set_ylim(altura, 0)
    eixo.text(
        0.2,
        0.42,
        "Cadeiras de deputado federal: Brasil, sob cada regra, com os votos de 2022",
        color=COR_TEXTO,
        fontsize=17,
        fontweight="bold",
    )

    for indice, cenario in enumerate(cenarios):
        origem_x = 0.25 + (indice % colunas_grade) * largura_bloco
        origem_y = 0.7 + (indice // colunas_grade) * altura_bloco
        tabela = resultados[resultados["cenario"] == cenario["id"]]
        total = int(tabela["total"].sum())
        eixo.text(
            origem_x,
            origem_y + 0.1,
            cenario["rotulo"],
            color=COR_TEXTO,
            fontsize=12,
            fontweight="bold",
            va="top",
        )
        eixo.text(
            origem_x,
            origem_y + 0.34,
            f"{total} cadeiras",
            color=COR_TEXTO_SECUNDARIO,
            fontsize=11,
            va="top",
        )
        posicao = 0
        for linha in tabela[tabela["total"] > 0].itertuples():
            ganhos = max(0, int(linha.variacao_vs_2022))
            for cadeira in range(int(linha.total)):
                excedente = cadeira >= linha.total - linha.excedente
                ganho = cadeira >= linha.total - ganhos
                marcador = "s" if excedente else "o"
                x = origem_x + 0.08 + (posicao % MAX_BOLINHAS_POR_LINHA) * passo
                y = origem_y + 0.7 + (posicao // MAX_BOLINHAS_POR_LINHA) * passo
                if ganho:
                    eixo.scatter(
                        x, y, s=100, marker=marcador, facecolor="none",
                        edgecolor=COR_ANEL, linewidth=1.1, zorder=2,
                    )
                eixo.scatter(
                    x, y, s=62, marker=marcador, color=cores[linha.ag],
                    edgecolor=COR_FUNDO, linewidth=0.5, zorder=3,
                )
                posicao += 1
        if indice:
            eixo.text(
                origem_x,
                origem_y + 0.75 + linhas_bloco * passo,
                textwrap.fill(_formatar_variacoes(tabela), width=78),
                color=COR_TEXTO_SECUNDARIO,
                fontsize=8.5,
                va="top",
            )

    legenda = [
        Line2D([], [], marker="o", linestyle="", color=cores[ag], markersize=11, label=ag)
        for ag in presentes
    ]
    legenda += [
        Line2D(
            [], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=COR_ANEL,
            markeredgewidth=1.6, markersize=12, label="Cadeira ganha em relação a 2022",
        ),
        Line2D(
            [], [], marker="s", linestyle="", color=COR_TEXTO_SECUNDARIO, markersize=10,
            label="Cadeira excedente",
        ),
    ]
    figura.legend(
        handles=legenda,
        loc="lower left",
        bbox_to_anchor=(0.2 / largura, 0.95 / altura),
        ncol=8,
        frameon=False,
        fontsize=10,
        labelcolor=COR_TEXTO,
        handletextpad=0.3,
        columnspacing=1.4,
    )
    notas = NOTAS.format(
        nao_inteiros="os grandes demais para um único distrito",
        ressalva="Soma dos 27 estados, cada um com seu próprio mapa distrital; outros mapas "
        "podem mudar os vencedores.",
    )
    figura.text(
        0.2 / largura,
        0.2 / altura,
        textwrap.fill(notas, width=int(largura * 11.5)),
        color=COR_TEXTO_SECUNDARIO,
        fontsize=9,
        va="bottom",
    )
    figura.savefig(f"{caminho_base}.png", dpi=300, metadata={"Software": None})
    with plt.rc_context({"svg.hashsalt": "pec-distrital"}):
        figura.savefig(f"{caminho_base}.svg", metadata={"Date": None, "Creator": None})
    plt.close(figura)


NOTA_ENSEMBLE = (
    "Notas: contrafactual mecânico com os votos de 2022. Cada amostra é um mapa válido gerado por "
    "ReCom, com as mesmas regras do mapa de base (contiguidade, ±20% do eleitorado, municípios "
    "inteiros salvo os grandes demais), e {n_amostras} amostras por número de distritos. A barra "
    "vai do percentil 5 ao 95; a linha fina, do mínimo ao máximo. A variação mede o efeito do "
    "desenho dos distritos, não a incerteza do voto. {extra}"
)

NOTA_METODOS = (
    "Notas: contrafactual mecânico com os votos de 2022, nos mesmos mapas de base em todas as "
    "colunas: só muda o método que calcula o direito proporcional de cada agremiação. Regra atual: "
    "quocientes e sobras por maior média. D'Hondt: divisores 1, 2, 3... Sainte-Laguë: divisores "
    "1, 3, 5... Barreira = 80% do quociente eleitoral. Excedentes sem limite. "
    "Cadeira ganha e saldo são medidos contra o resultado real de 2022."
)


def _gravar(figura, caminho_base: Path) -> None:
    figura.savefig(f"{caminho_base}.png", dpi=300, metadata={"Software": None})
    with plt.rc_context({"svg.hashsalt": "pec-distrital"}):
        figura.savefig(f"{caminho_base}.svg", metadata={"Date": None, "Creator": None})
    plt.close(figura)


def desenhar_figura_ensemble(
    resumo: pd.DataFrame,
    estado: pd.DataFrame,
    cenarios: list[dict],
    cores: dict[str, str],
    nome: str,
    notas: str,
    caminho_base: Path,
) -> None:
    """Faixa de cadeiras por agremiação em cada cenário PEC, com o mapa de base marcado."""
    cenarios_pec = [cenario for cenario in cenarios if cenario["tipo"] == "pec"]
    presentes = list(resumo.loc[resumo["total_max"] > 0, "ag"].drop_duplicates())
    sem_cor = [ag for ag in presentes if ag not in cores]
    if sem_cor:
        raise ValueError(f"Agremiações sem cor em cores_agremiacoes.csv: {sem_cor}")

    colunas_grade = 2
    linhas_grade = math.ceil(len(cenarios_pec) / colunas_grade)
    altura_painel = 0.8 + 0.27 * len(presentes)
    largura = 14.0
    rodape = 1.5
    altura = altura_painel * linhas_grade + rodape + 1.15
    figura, eixos = plt.subplots(
        linhas_grade,
        colunas_grade,
        figsize=(largura, altura),
        sharex=True,
        squeeze=False,
        facecolor=COR_FUNDO,
        gridspec_kw={
            "left": 0.13,
            "right": 0.985,
            "top": 1 - 1.15 / altura,
            "bottom": rodape / altura,
            "wspace": 0.2,
            "hspace": 0.3,
        },
    )
    limite = float(resumo["total_max"].max()) + 0.5
    for eixo, cenario in zip(eixos.ravel(), cenarios_pec):
        tabela = resumo[resumo["cenario"] == cenario["id"]].set_index("ag").reindex(presentes)
        tamanho = estado[estado["cenario"] == cenario["id"]].iloc[0]
        for y, (ag, linha) in enumerate(tabela.iterrows()):
            eixo.plot([linha["total_min"], linha["total_max"]], [y, y], color="#AFB8C1", lw=1.2, zorder=1)
            eixo.plot(
                [linha["total_p5"], linha["total_p95"]],
                [y, y],
                color=cores[ag],
                lw=6,
                solid_capstyle="round",
                zorder=2,
            )
            eixo.scatter(
                linha["total_mediana"], y, s=44, color=cores[ag], edgecolor=COR_FUNDO, linewidth=1.2, zorder=3
            )
            if pd.notna(linha.get("total_mapa_base")):
                eixo.scatter(
                    linha["total_mapa_base"],
                    y,
                    s=46,
                    marker="D",
                    facecolor="none",
                    edgecolor=COR_ANEL,
                    linewidth=1.2,
                    zorder=4,
                )
        eixo.set_yticks(range(len(presentes)), presentes, fontsize=9.5, color=COR_TEXTO)
        eixo.set_ylim(len(presentes) - 0.5, -0.5)
        eixo.set_xlim(-0.5, limite)
        eixo.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        eixo.tick_params(axis="x", labelbottom=True, labelsize=9, colors=COR_TEXTO_SECUNDARIO)
        eixo.tick_params(axis="y", length=0)
        eixo.grid(axis="x", color="#EAEEF2", lw=0.8)
        eixo.set_axisbelow(True)
        for lado in ("top", "right", "left"):
            eixo.spines[lado].set_visible(False)
        eixo.spines["bottom"].set_color("#D0D7DE")
        eixo.set_title(
            f"{cenario['rotulo']}\nTotal: {int(tamanho['total_mediana'])} cadeiras "
            f"(de {int(tamanho['total_min'])} a {int(tamanho['total_max'])})",
            loc="left",
            fontsize=10.5,
            color=COR_TEXTO,
        )
    for eixo in eixos.ravel()[len(cenarios_pec):]:
        eixo.set_visible(False)

    figura.text(
        0.012,
        1 - 0.42 / altura,
        f"Quanto o desenho dos distritos muda as cadeiras: {nome}, com os votos de 2022",
        color=COR_TEXTO,
        fontsize=14,
        fontweight="bold",
    )
    legenda = [
        Line2D([], [], color=COR_TEXTO_SECUNDARIO, lw=6, label="Percentis 5 a 95"),
        Line2D([], [], color="#AFB8C1", lw=1.2, label="Mínimo a máximo"),
        Line2D([], [], marker="o", linestyle="", color=COR_TEXTO_SECUNDARIO, markersize=7, label="Mediana"),
        Line2D(
            [], [], marker="D", linestyle="", markerfacecolor="none", markeredgecolor=COR_ANEL,
            markersize=7, label="Mapa de base",
        ),
    ]
    figura.legend(
        handles=legenda,
        loc="lower left",
        bbox_to_anchor=(0.012, 0.85 / altura),
        ncol=4,
        frameon=False,
        fontsize=10,
        labelcolor=COR_TEXTO,
    )
    figura.text(
        0.012,
        0.12 / altura,
        textwrap.fill(notas, width=165),
        color=COR_TEXTO_SECUNDARIO,
        fontsize=9,
        va="bottom",
    )
    _gravar(figura, caminho_base)


def _desenhar_bloco(eixo, tabela, cores, x0, y0, passo, tamanho) -> None:
    """Bolinhas de um cenário, com no máximo MAX_BOLINHAS_POR_LINHA por linha."""
    posicao = 0
    for linha in tabela[tabela["total"] > 0].itertuples():
        ganhos = max(0, int(linha.variacao_vs_2022))
        for cadeira in range(int(linha.total)):
            excedente = cadeira >= linha.total - linha.excedente
            ganho = cadeira >= linha.total - ganhos
            marcador = "s" if excedente else "o"
            x = x0 + (posicao % MAX_BOLINHAS_POR_LINHA) * passo
            y = y0 + (posicao // MAX_BOLINHAS_POR_LINHA) * passo
            if ganho:
                eixo.scatter(
                    x, y, s=tamanho * 1.6, marker=marcador, facecolor="none",
                    edgecolor=COR_ANEL, linewidth=1.1, zorder=2,
                )
            eixo.scatter(
                x, y, s=tamanho, marker=marcador, color=cores[linha.ag],
                edgecolor=COR_FUNDO, linewidth=0.5, zorder=3,
            )
            posicao += 1


def desenhar_figura_metodos(
    resultados: pd.DataFrame,
    linhas: list[dict],
    colunas: list[dict],
    real_id: str,
    cores: dict[str, str],
    nome: str,
    caminho_base: Path,
) -> None:
    """Grade de bolinhas: uma linha por regra da PEC e uma coluna por método de alocação.

    Cada célula é o cenário de id `{linha}__{coluna}` de `resultados`.
    """
    presentes = list(resultados.loc[resultados["total"] > 0, "ag"].unique())
    sem_cor = [ag for ag in presentes if ag not in cores]
    if sem_cor:
        raise ValueError(f"Agremiações sem cor em cores_agremiacoes.csv: {sem_cor}")

    totais = resultados.groupby("cenario")["total"].sum()
    largura_pontos = min(int(totais.max()), MAX_BOLINHAS_POR_LINHA)
    passo = 0.2
    linhas_bolinhas = contar_linhas(int(totais.max()))
    largura_bloco = max(largura_pontos * passo, 3.4) + 0.45
    altura_bloco = linhas_bolinhas * passo + 1.05
    margem_esquerda = 1.75
    topo = 1.05
    real = resultados[resultados["cenario"] == real_id]
    altura_real = linhas_bolinhas * passo + 0.75
    rodape = 1.75
    largura = margem_esquerda + largura_bloco * len(colunas) + 0.15
    altura = topo + altura_real + altura_bloco * len(linhas) + rodape

    figura = plt.figure(figsize=(largura, altura), facecolor=COR_FUNDO)
    eixo = figura.add_axes([0, 0, 1, 1])
    eixo.set_axis_off()
    eixo.set_xlim(0, largura)
    eixo.set_ylim(altura, 0)
    eixo.text(
        0.15,
        0.38,
        f"Cadeiras de deputado federal por método de alocação: {nome}, com os votos de 2022",
        color=COR_TEXTO,
        fontsize=14,
        fontweight="bold",
    )

    y_real = topo - 0.3
    eixo.text(0.15, y_real, "Resultado real\nde 2022", color=COR_TEXTO, fontsize=10.5, va="top")
    _desenhar_bloco(eixo, real, cores, margem_esquerda + 0.08, y_real + 0.1, passo, 62)
    composicao = " · ".join(f"{l.ag} {l.total}" for l in real[real["total"] > 0].itertuples())
    eixo.text(
        margem_esquerda,
        y_real + 0.1 + (linhas_bolinhas - 0.4) * passo,
        f"{int(real['total'].sum())} cadeiras: {composicao}",
        color=COR_TEXTO_SECUNDARIO,
        fontsize=8.5,
        va="top",
    )

    y_grade = topo + altura_real
    for j, coluna in enumerate(colunas):
        eixo.text(
            margem_esquerda + j * largura_bloco,
            y_grade - 0.02,
            coluna["rotulo"],
            color=COR_TEXTO,
            fontsize=11.5,
            fontweight="bold",
            va="bottom",
        )
    caracteres = int((largura_bloco - 0.2) * 14.5)
    for i, linha in enumerate(linhas):
        y = y_grade + i * altura_bloco
        eixo.text(0.15, y + 0.08, textwrap.fill(linha["rotulo"], 16), color=COR_TEXTO, fontsize=10.5, va="top")
        for j, coluna in enumerate(colunas):
            x = margem_esquerda + j * largura_bloco
            tabela = resultados[resultados["cenario"] == f"{linha['id']}__{coluna['id']}"]
            total = int(tabela["total"].sum())
            eixo.text(x, y + 0.08, f"{total} cadeiras", color=COR_TEXTO, fontsize=9.5, fontweight="bold", va="top")
            _desenhar_bloco(eixo, tabela, cores, x + 0.08, y + 0.4, passo, 62)
            eixo.text(
                x,
                y + 0.27 + linhas_bolinhas * passo,
                textwrap.fill(_formatar_variacoes(tabela), caracteres),
                color=COR_TEXTO_SECUNDARIO,
                fontsize=8,
                va="top",
            )

    legenda = [
        Line2D([], [], marker="o", linestyle="", color=cores[ag], markersize=10, label=ag)
        for ag in presentes
    ]
    legenda += [
        Line2D(
            [], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=COR_ANEL,
            markeredgewidth=1.6, markersize=11, label="Cadeira ganha em relação a 2022",
        ),
        Line2D(
            [], [], marker="s", linestyle="", color=COR_TEXTO_SECUNDARIO, markersize=9,
            label="Cadeira excedente",
        ),
    ]
    figura.legend(
        handles=legenda,
        loc="lower left",
        bbox_to_anchor=(0.15 / largura, 0.72 / altura),
        ncol=max(4, int(largura / 1.45)),
        frameon=False,
        fontsize=9.5,
        labelcolor=COR_TEXTO,
        handletextpad=0.3,
        columnspacing=1.2,
    )
    figura.text(
        0.15 / largura,
        0.12 / altura,
        textwrap.fill(NOTA_METODOS, width=int(largura * 12)),
        color=COR_TEXTO_SECUNDARIO,
        fontsize=8.5,
        va="bottom",
    )
    _gravar(figura, caminho_base)


def desenhar_mapa_distritos(
    geometrias: gpd.GeoDataFrame,
    rotulos: dict[int, str],
    secoes: pd.DataFrame,
    distritos_zoom: list[int],
    caminho: Path,
) -> None:
    """Estado à esquerda e maior município dividido à direita, com os locais de votação."""
    locais = secoes.drop_duplicates(["NR_LATITUDE", "NR_LONGITUDE"])
    pontos = gpd.GeoDataFrame(
        locais,
        geometry=gpd.points_from_xy(locais["NR_LONGITUDE"], locais["NR_LATITUDE"]),
        crs="EPSG:4674",
    ).to_crs(geometrias.crs)
    paleta = CORES_DISTRITOS + [
        matplotlib.colors.to_hex(cor) for cor in matplotlib.colormaps["tab20"].colors
    ]
    cores = {
        distrito: paleta[(distrito - 1) % len(paleta)] for distrito in geometrias["distrito"]
    }
    muitos = len(rotulos) > 12
    tamanho_ponto = 0.6 if muitos else 4

    figura, (eixo_estado, eixo_zoom) = plt.subplots(
        1, 2, figsize=(14, 7.5), width_ratios=[1.6, 1]
    )
    for eixo in (eixo_estado, eixo_zoom):
        geometrias.plot(
            ax=eixo,
            color=[cores[distrito] for distrito in geometrias["distrito"]],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.75,
        )
        eixo.set_axis_off()
        eixo.set_aspect("equal")

    pontos.plot(ax=eixo_estado, color="#222222", markersize=tamanho_ponto)
    pontos.plot(ax=eixo_zoom, color="#222222", markersize=tamanho_ponto * 1.5)

    zoom = geometrias[geometrias["distrito"].isin(distritos_zoom)]
    limites = (zoom if not zoom.empty else geometrias).total_bounds
    eixo_zoom.set_xlim(limites[0], limites[2])
    eixo_zoom.set_ylim(limites[1], limites[3])
    eixo_estado.set_title("Estado")
    eixo_zoom.set_title("Município dividido")

    if not muitos:
        legenda = [
            Patch(facecolor=cores[distrito], alpha=0.75, label=f"{distrito}: {rotulos[distrito]}")
            for distrito in sorted(rotulos)
        ]
        figura.legend(handles=legenda, loc="lower center", ncol=1, fontsize=8, frameon=False)
    figura.suptitle(f"Mapa de diagnóstico: {len(rotulos)} distritos")
    figura.subplots_adjust(bottom=0.05 if muitos else 0.2, top=0.92, left=0.02, right=0.98)
    figura.savefig(caminho, dpi=200, metadata={"Software": None})
    plt.close(figura)
