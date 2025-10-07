import os

import joblib
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset


def executar_pipeline_treinamento(
    avaliacoes_df: pd.DataFrame, cachacas_df: pd.DataFrame
):
    """
    Recebe os DataFrames de avaliações e cachaças, treina o modelo de recomendação
    e salva os artefatos necessários para uso posterior.

    Args:
        avaliacoes_df (pd.DataFrame): DataFrame com colunas ['user.id', 'cachaca.id', 'notaGeral'].
        cachacas_df (pd.DataFrame): DataFrame com colunas ['id', 'nome', 'tipoCachaca', 'regiao'].
    """

    # --- Validação Inicial ---
    if avaliacoes_df.empty or cachacas_df.empty:
        print("Erro: Os DataFrames de entrada não podem estar vazios.")
        return

    print("✅ INICIANDO O PIPELINE DE TREINAMENTO DA IA.")

    # Diretório para salvar o modelo treinado e outros artefatos
    ARTIFACTS_PATH = "artifacts/"
    if not os.path.exists(ARTIFACTS_PATH):
        os.makedirs(ARTIFACTS_PATH)

    # ======================================================================================
    # PASSO 1: Preparar o "Dataset" do LightFM
    # Objetivo: Informar ao LightFM todos os usuários, itens e características (features)
    # que existem no nosso sistema. Ele criará um "dicionário" interno para mapear tudo.
    # ======================================================================================
    print("\nPASSO 1: Mapeando usuários, itens e features...")
    dataset = Dataset()
    dataset.fit(
        users=avaliacoes_df["user.id"].unique(),
        items=cachacas_df["id"].unique(),
        item_features=cachacas_df["tipoCachaca"].unique().tolist()
        + cachacas_df["regiao"].unique().tolist(),
    )
    print("Mapeamento concluído.")

    # ======================================================================================
    # PASSO 2: Construir a Matriz de Interações (Usuário-Item)
    # Objetivo: Criar a principal estrutura de dados que diz "qual usuário avaliou qual
    # cachaça e com que nota". Esta matriz é a base da Filtragem Colaborativa.
    # ======================================================================================
    print("\nPASSO 2: Construindo a matriz de interações (usuário x cachaça)...")
    (interactions, weights) = dataset.build_interactions(
        # Fornecemos uma tupla para cada avaliação no formato (id_usuario, id_item, peso)
        (row["user.id"], row["cachaca.id"], row["notaGeral"])
        for _, row in avaliacoes_df.iterrows()
    )
    print("Matriz de interações construída.")

    # ======================================================================================
    # PASSO 3: Construir a Matriz de Características dos Itens
    # Objetivo: Criar uma estrutura que descreve cada cachaça por suas características
    # (ex: 'tipo: Ouro', 'região: Salinas'). Esta matriz é a base da Filtragem
    # Baseada em Conteúdo.
    # ======================================================================================
    print("\nPASSO 3: Construindo a matriz de características das cachaças...")
    item_features = dataset.build_item_features(
        # Fornecemos uma tupla para cada cachaça no formato (id_item, [lista_de_features])
        (row["id"], [row["tipoCachaca"], row["regiao"]])
        for _, row in cachacas_df.iterrows()
    )
    print("Matriz de características construída.")

    # ======================================================================================
    # PASSO 4: Instanciar e Treinar o Modelo
    # Objetivo: Alimentar o modelo LightFM com as matrizes criadas para que ele aprenda os
    # padrões de gosto dos usuários.
    # ======================================================================================
    print("\nPASSO 4: Instanciando e treinando o modelo LightFM...")

    # Usamos 'warp' (Weighted Approximate-Rank Pairwise) porque ele é ótimo para otimizar
    # a ordem (ranking) das recomendações, que é exatamente o que queremos.
    model = LightFM(loss="warp", random_state=42, no_components=30, learning_rate=0.05)

    # O método 'fit' inicia o treinamento.
    model.fit(
        interactions,  # A matriz de quem avaliou o quê
        item_features=item_features,  # As características de cada cachaça
        sample_weight=weights,  # As notas dadas em cada avaliação
        epochs=20,  # Número de vezes que o modelo "estuda" os dados
        num_threads=4,  # Quantos processadores usar para acelerar
        verbose=True,  # Mostra o progresso do treinamento
    )
    print("Treinamento concluído.")

    # ======================================================================================
    # PASSO 5: Salvar os Artefatos
    # Objetivo: Guardar o resultado do nosso trabalho. Salvamos o modelo treinado e os
    # mapeamentos para que o script de recomendação possa usá-los sem precisar
    # treinar tudo de novo.
    # ======================================================================================
    print("\nPASSO 5: Salvando o modelo treinado e os artefatos...")

    # Usamos joblib pois é eficiente para salvar objetos Python complexos
    joblib.dump(model, os.path.join(ARTIFACTS_PATH, "model.pkl"))
    joblib.dump(dataset, os.path.join(ARTIFACTS_PATH, "dataset.pkl"))

    print(
        f"✅ PIPELINE DE TREINAMENTO CONCLUÍDO! Artefatos salvos em '{ARTIFACTS_PATH}'."
    )


# --- BLOCO DE EXECUÇÃO DE EXEMPLO ---
if __name__ == "__main__":

    print("Iniciando simulação com dados de exemplo...\n")
    # Simulação: Carregando dados como se viessem do banco de dados

    # DataFrame de Avaliações
    avaliacoes_exemplo_df = pd.DataFrame(
        {
            "user.id": [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
            "cachaca.id": [101, 103, 101, 102, 102, 103, 104, 101, 102, 103, 104],
            "notaGeral": [9, 8, 10, 7, 5, 9, 8, 10, 6, 8, 7],
        }
    )

    # DataFrame de Cachaças
    cachacas_exemplo_df = pd.DataFrame(
        {
            "id": [101, 102, 103, 104, 105],
            "nome": [
                "Serra Limpa",
                "Salineira",
                "Rainha do Vale",
                "Encantos da Marquesa",
                "Vale Verde",
            ],
            "tipoCachaca": ["BRANCA", "OURO", "ENVELHECIDA", "BRANCA", "ENVELHECIDA"],
            "regiao": ["Paraíba", "Salinas", "Salinas", "Paraíba", "Minas Gerais"],
        }
    )

    # Chama a função principal do pipeline com os dados de exemplo
    executar_pipeline_treinamento(avaliacoes_exemplo_df, cachacas_exemplo_df)
