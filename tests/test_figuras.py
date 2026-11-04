"""Testes da figura de bolinhas (seção 6)."""

import pandas as pd
import pytest

import figuras


@pytest.mark.parametrize(
    ("n_bolinhas", "esperado"), [(8, 1), (30, 1), (31, 2), (60, 2), (70, 3), (72, 3)]
)
def test_cada_linha_tem_no_maximo_30_bolinhas(n_bolinhas, esperado):
    assert figuras.contar_linhas(n_bolinhas) == esperado


def test_figura_grava_png_e_svg_e_exige_cor(tmp_path):
    resultados = pd.DataFrame(
        {
            "cenario": ["real", "pec"],
            "ag": ["A", "A"],
            "excedente": [0, 2],
            "total": [70, 72],
            "variacao_vs_real": [0, 2],
        }
    )
    cenarios = [{"id": "real", "rotulo": "Real"}, {"id": "pec", "rotulo": "PEC"}]
    base = tmp_path / "figura"
    figuras.desenhar_figura_cenarios(resultados, cenarios, {"A": "#123456"}, "SP", "SP", base)
    assert base.with_suffix(".png").stat().st_size > 0
    assert base.with_suffix(".svg").stat().st_size > 0
    with pytest.raises(ValueError, match="sem cor"):
        figuras.desenhar_figura_cenarios(resultados, cenarios, {}, "SP", "SP", base)


def test_resumo_separa_o_que_ja_esta_no_tse_do_que_falta():
    tabela = pd.DataFrame(
        {
            "ano": [2022, 2022, 2022, 2018],
            "uf": ["AP", "CE", "AL", "SE"],
            "n_cadeiras": [4, 1, 1, 1],
            "na_totalizacao": ["sim", "sim", "nao", "sim"],
        }
    )
    texto = figuras.resumir_diferencas(tabela, 2022)
    assert "já inclui 5 cadeiras" in texto and "difere da Câmara em 1 cadeira (AL)" in texto
    assert "não tem outra diferença conhecida" in figuras.resumir_diferencas(tabela, 2022, "AP")
    assert "já inclui 1 cadeira trocada de" in figuras.resumir_diferencas(tabela, 2018, "SE")
    assert figuras.resumir_diferencas(tabela, 2018, "AP") == ""


def test_figura_do_conjunto_mostra_partido_que_so_tem_cadeira_real(tmp_path):
    from ensemble import anexar_resultado_real
    from figuras import NOTA_ENSEMBLE, desenhar_figura_ensemble

    colunas = dict.fromkeys(
        ["total_min", "total_p5", "total_mediana", "total_p95", "total_max"], 0
    )
    resumo = pd.DataFrame(
        [
            {"cenario": "pec", "ag": "A", **{c: 5 for c in colunas}, "total_mapa_base": 5},
            {"cenario": "pec", "ag": "B", **colunas, "total_mapa_base": 0},
            {"cenario": "pec", "ag": "C", **colunas, "total_mapa_base": 0},
        ]
    )
    resultados = pd.DataFrame(
        {"cenario": ["real_2018", "real_2018", "pec"], "ag": ["A", "B", "A"], "total": [4, 9, 5]}
    )
    resumo = anexar_resultado_real(resumo, resultados)
    assert resumo["total_real"].tolist() == [4, 9, 0]

    estado = pd.DataFrame(
        {"cenario": ["pec"], "total_mediana": [5], "total_min": [5], "total_max": [5]}
    )
    nota = NOTA_ENSEMBLE.format(ano=2018, n_amostras=10, extra="")
    assert "traço preto" in nota
    cenarios = [{"id": "pec", "tipo": "pec", "rotulo": "PEC"}]
    base = tmp_path / "figura"
    desenhar_figura_ensemble(resumo, estado, cenarios, {"A": "#000", "B": "#111"}, "X", nota, base)
    assert base.with_suffix(".png").exists()


def test_resultado_real_recusa_partido_fora_do_resumo():
    from ensemble import anexar_resultado_real

    resumo = pd.DataFrame({"cenario": ["pec"], "ag": ["A"]})
    resultados = pd.DataFrame({"cenario": ["real_2018"], "ag": ["Z"], "total": [1]})
    with pytest.raises(ValueError, match="Z"):
        anexar_resultado_real(resumo, resultados)
