# data_fetcher.py
import os
from typing import Dict, Optional

import pandas as pd
import requests


def _get_api_headers() -> Dict[str, str]:
    """Retorna os headers de autenticação para a API."""
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("A variável de ambiente API_KEY não foi definida.")
    return {"Authorization": f"Bearer {api_key}"}


def get_data_from_api(endpoint: str) -> Optional[pd.DataFrame]:
    """
    Busca dados de um endpoint da API e os retorna como um DataFrame do Pandas.

    Args:
        endpoint: O caminho do endpoint (ex: '/avaliacoes').

    Returns:
        Um DataFrame com os dados ou None em caso de erro.
    """
    base_url = os.getenv("API_BASE_URL")
    url = f"{base_url}{endpoint}"

    print(f"Buscando dados de: {url}")
    try:
        response = requests.get(url, headers=_get_api_headers())
        response.raise_for_status()  # Lança um erro para status HTTP 4xx/5xx

        data = response.json()
        if not data:
            print(f"Aviso: Nenhum dado retornado do endpoint {endpoint}.")
            return pd.DataFrame()

        # Normaliza o JSON para DataFrame, especialmente se os dados forem aninhados
        return pd.json_normalize(data)

    except requests.exceptions.RequestException as e:
        print(f"Erro ao buscar dados da API em {url}: {e}")
        return None


if __name__ == "__main__":
    # Teste rápido do módulo (requer .env configurado)
    from dotenv import load_dotenv

    load_dotenv()

    avaliacoes = get_data_from_api("/avaliacoes")
    if avaliacoes is not None:
        print("\nAmostra de Avaliações:")
        print(avaliacoes.head())

    cachacas = get_data_from_api("/cachacas")
    if cachacas is not None:
        print("\nAmostra de Cachaças:")
        print(cachacas.head())
