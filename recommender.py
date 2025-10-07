# recommender.py
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

# Define o caminho padrão para os artefatos salvos pelo model_trainer
ARTIFACTS_PATH = "artifacts/"


class Recommender:
    def __init__(self):
        """
        Carrega todos os artefatos necessários (modelo, dataset, etc.)
        quando uma instância da classe é criada.
        """
        self.model = None
        self.dataset = None
        self.cachacas_df = None
        self.user_id_map = None
        self.item_id_map_inv = None

        try:
            print("Carregando artefatos do modelo treinado...")
            self.model = joblib.load(ARTIFACTS_PATH + "model.pkl")
            self.dataset = joblib.load(ARTIFACTS_PATH + "dataset.pkl")
            self.cachacas_df = joblib.load(ARTIFACTS_PATH + "cachacas_df.pkl")

            # Extrai os mapeamentos do dataset para traduzir IDs
            # user_id_map: Converte o ID real do usuário para o índice interno do modelo
            # item_id_map: Converte o ID real da cachaça para o índice interno
            user_id_map, _, item_id_map, _ = self.dataset.mapping()
            self.user_id_map = user_id_map

            # Criamos um mapeamento inverso para converter o índice interno de volta para o ID real
            self.item_id_map_inv = {v: k for k, v in item_id_map.items()}

            print("Artefatos carregados com sucesso.")

        except FileNotFoundError:
            print("ERRO: Arquivos de modelo não encontrados no diretório 'artifacts/'.")
            print("Por favor, execute o 'model_trainer.py' primeiro.")

    def generate_recommendations(
        self, user_id: Any, user_ratings_df: pd.DataFrame, top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Gera uma lista das N melhores recomendações para um usuário específico.

        Args:
            user_id: O ID do usuário para o qual gerar recomendações.
            user_ratings_df: DataFrame contendo todas as avaliações para filtrar itens já vistos.
            top_n: O número de recomendações a serem retornadas.

        Returns:
            Uma lista de dicionários, onde cada dicionário contém os detalhes de uma cachaça recomendada.
        """
        if not self.model or not self.user_id_map:
            print("Recomendador não foi inicializado corretamente. Abortando.")
            return []

        # Verifica se o usuário existe no nosso dataset de treinamento
        if user_id not in self.user_id_map:
            print(
                f"Usuário {user_id} é novo ou não possui avaliações. Não é possível gerar recomendações personalizadas."
            )
            return []

        # Pega o índice interno do usuário, que o modelo entende
        internal_user_id = self.user_id_map[user_id]

        # Pega a lista de todas as cachaças que o usuário já avaliou para não recomendá-las novamente
        known_positives_ids = user_ratings_df[user_ratings_df["user.id"] == user_id][
            "cachaca.id"
        ].tolist()

        # Pega os índices de todos os itens possíveis em nosso catálogo
        all_item_indices = np.arange(len(self.item_id_map_inv))

        # O CORAÇÃO DA RECOMENDAÇÃO:
        # O modelo calcula um "score" para cada cachaça para este usuário específico.
        scores = self.model.predict(
            internal_user_id,
            all_item_indices,
            item_features=self.dataset.build_item_features(
                (
                    (row["id"], [row["tipoCachaca"], row["regiao"]])
                    for _, row in self.cachacas_df.iterrows()
                )
            ),
        )

        # Ordena os itens pelo score (do maior para o menor)
        top_items_indices = np.argsort(-scores)

        recommendations = []
        for item_index in top_items_indices:
            # Converte o índice interno de volta para o ID original da cachaça
            original_item_id = self.item_id_map_inv[item_index]

            # Adiciona à lista de recomendação apenas se o usuário ainda não avaliou
            if original_item_id not in known_positives_ids:
                item_details = self.cachacas_df[
                    self.cachacas_df["id"] == original_item_id
                ]
                if not item_details.empty:
                    recommendations.append(item_details.to_dict("records")[0])

            # Para quando atingimos o número desejado de recomendações
            if len(recommendations) >= top_n:
                break

        return recommendations
