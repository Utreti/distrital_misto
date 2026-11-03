"""Dados reais do Acre 2022, carregados uma única vez por sessão de testes."""

import pytest

import carga


@pytest.fixture(scope="session")
def params():
    return carga.ler_params()


@pytest.fixture(scope="session")
def dados(params):
    return carga.carregar_dados(params)


@pytest.fixture(scope="session")
def votos_ag(dados):
    return dados["votos_ag"].set_index("agremiacao")["votos"]
