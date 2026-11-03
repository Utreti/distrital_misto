"""Testes da Fase 3: apuração vetorizada e resumo das amostras."""

import resource
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import distritos
import ensemble
import vencedores


class _Proposta(dict):
    pass


def _alocar(n_bytes):
    return len(bytearray(n_bytes))


def _teto_de_dados():
    return resource.getrlimit(resource.RLIMIT_DATA)[0]


def _votos(unidades, candidatos, votos, agremiacao, ranking=None):
    return ensemble.Votos(
        unidades=np.array(unidades),
        candidatos=np.array(candidatos),
        votos=np.array(votos, dtype="float64"),
        n_candidatos=len(agremiacao),
        ranking=np.array(ranking if ranking is not None else range(len(agremiacao))),
        agremiacao=np.array(agremiacao),
        n_agremiacoes=max(agremiacao) + 1,
    )


def test_candidato_vence_apenas_seu_distrito_mais_forte():
    votos = _votos([0, 1, 0, 1], [0, 0, 1, 1], [100, 90, 50, 40], [0, 1])
    por_ag, ganhadores = ensemble.apurar_distritos(np.array([0, 1]), votos, 2)
    assert por_ag.tolist() == [1, 1]
    assert ganhadores.tolist() == [0, 1]


def test_empate_vai_para_o_melhor_no_desempate():
    votos = _votos([0, 0], [0, 1], [100, 100], [0, 1], ranking=[1, 0])
    _, ganhadores = ensemble.apurar_distritos(np.array([0]), votos, 1)
    assert ganhadores.tolist() == [1]


def test_distrito_sem_vencedor_gera_erro():
    votos = _votos([0], [0], [100], [0])
    with pytest.raises(ValueError, match="sem vencedor"):
        ensemble.apurar_distritos(np.array([0, 1]), votos, 2)


def test_apuracao_vetorizada_coincide_com_a_regra_do_mapa_de_base():
    rng = np.random.default_rng(7)
    n_unidades, n_candidatos, n_distritos = 40, 30, 6
    linhas = [
        (u, c, int(rng.integers(1, 500)))
        for u in range(n_unidades)
        for c in range(n_candidatos)
        if rng.random() < 0.5
    ]
    unidades, cand, valores = map(list, zip(*linhas))
    agremiacao = rng.integers(0, 5, n_candidatos)
    votos = _votos(unidades, cand, valores, agremiacao.tolist())
    distrito_da_unidade = rng.integers(0, n_distritos, n_unidades)
    distrito_da_unidade[:n_distritos] = range(n_distritos)
    _, ganhadores = ensemble.apurar_distritos(distrito_da_unidade, votos, n_distritos)

    pares = pd.DataFrame({"distrito": distrito_da_unidade[unidades], "SQ_CANDIDATO": cand, "votos": valores})
    pares = pares.groupby(["distrito", "SQ_CANDIDATO"], as_index=False)["votos"].sum()
    pares = pares.assign(agremiacao="X", SQ_CANDIDATO=pares["SQ_CANDIDATO"].astype(str))
    cadastro = pd.DataFrame(
        {
            "SQ_CANDIDATO": [str(c) for c in range(n_candidatos)],
            "NM_URNA_CANDIDATO": "x",
            "DT_NASCIMENTO": pd.date_range("1950-01-01", periods=n_candidatos),
        }
    )
    esperado = vencedores.apurar_vencedores(pares, cadastro)["SQ_CANDIDATO"].astype(int)
    assert ganhadores.tolist() == esperado.tolist()


def test_gelman_rubin_detecta_cadeias_que_nao_se_misturam():
    misturadas = np.tile([0, 1, 0, 1], 2)
    separadas = np.array([0, 1, 0, 1, 5, 6, 5, 6])
    assert ensemble.calcular_gelman_rubin(misturadas, 2) < 1.05
    assert ensemble.calcular_gelman_rubin(separadas, 2) > 2


def _cenario(proporcao="1/2", barreira=False):
    return {
        "id": "pec",
        "tipo": "pec",
        "proporcao_distrital": proporcao,
        "metodo": "quociente_atual",
        "barreira": barreira,
    }


def test_resumo_aplica_a_regra_compensatoria_em_cada_amostra():
    # Direito de A e B: 2 cadeiras cada, em 4; D de A varia entre 0 e 2 e nunca excede.
    votos = pd.Series({"A": 500, "B": 500})
    matriz = np.array([[2, 0], [1, 1], [0, 2]])
    por_ag, por_estado = ensemble.resumir_cenarios({2: matriz}, ["A", "B"], votos, [_cenario()], 4)
    assert por_ag["total_max"].tolist() == [2, 2]
    assert por_ag["distritos_max"].tolist() == [2, 2]
    assert por_estado["total_max"].tolist() == [4]
    assert por_estado["excedente_medio"].tolist() == [0]


def test_resumo_mede_excedente_e_passagem_de_70():
    votos = pd.Series({"A": 999, "B": 1})
    # B vence os dois distritos em uma amostra: tem direito a 0 e leva 2 excedentes.
    matriz = np.array([[2, 0], [0, 2]])
    _, por_estado = ensemble.resumir_cenarios({2: matriz}, ["A", "B"], votos, [_cenario()], 3)
    assert por_estado["total_min"].tolist() == [3]
    assert por_estado["total_max"].tolist() == [5]
    assert por_estado["excedente_medio"].tolist() == [1]
    assert por_estado["prob_acima_de_70"].tolist() == [0]


def test_soma_nacional_preserva_totais_dos_estados():
    votos = {"AA": pd.Series({"A": 500, "B": 500}), "BB": pd.Series({"A": 900, "B": 100})}
    amostras = {
        "AA": pd.DataFrame({"mapa": 1, "amostra": [0, 1], "ag": ["A", "B"], "distritos": 1}),
        "BB": pd.DataFrame({"mapa": 1, "amostra": [0, 1], "ag": ["A", "A"], "distritos": 1}),
    }
    por_ag, por_pais = ensemble.somar_brasil(amostras, votos, {"AA": 2, "BB": 2}, [_cenario()], 1)
    assert por_ag.set_index("ag")["cadeiras_E"].to_dict() == {"A": 3, "B": 1}
    assert por_pais["total_min"].tolist() == [4]


def test_passo_aceito_nao_guarda_a_linhagem_de_mapas():
    pai = SimpleNamespace(parent=object())
    proposta = _Proposta({"aptos": {0: 100, 1: 100}})
    problema = SimpleNamespace(max_tentativas=1, limites=(50, 150))
    regiao = SimpleNamespace(alvo=100, tolerancia=0.2)
    with patch.object(ensemble, "recom", return_value=proposta), patch.object(ensemble, "COLUNA_APTOS", "aptos"):
        novo, aceito = ensemble._dar_passo(pai, regiao, problema, None)
    assert aceito and novo is proposta
    assert pai.parent is None


def test_teto_total_de_memoria_e_dividido_entre_os_processos_filhos():
    distritos._MEMORIA_TOTAL_GB = 60
    try:
        with distritos.criar_piscina(4) as piscina:
            assert piscina.apply(_teto_de_dados) >= 15 * 1000**3
            with pytest.raises(MemoryError):
                piscina.apply(_alocar, (16 * 1000**3,))
    finally:
        distritos._MEMORIA_TOTAL_GB = None
