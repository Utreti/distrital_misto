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
            "variacao_vs_2022": [0, 2],
        }
    )
    cenarios = [{"id": "real", "rotulo": "Real"}, {"id": "pec", "rotulo": "PEC"}]
    base = tmp_path / "figura"
    figuras.desenhar_figura_cenarios(resultados, cenarios, {"A": "#123456"}, "SP", "SP", base)
    assert base.with_suffix(".png").stat().st_size > 0
    assert base.with_suffix(".svg").stat().st_size > 0
    with pytest.raises(ValueError, match="sem cor"):
        figuras.desenhar_figura_cenarios(resultados, cenarios, {}, "SP", "SP", base)
