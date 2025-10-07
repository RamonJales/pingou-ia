# main.py
import model_trainer
from dotenv import load_dotenv

import data_fetcher
import email_sender
from recommender import Recommender


def run_recommendation_pipeline():
    """
    Orquestra o pipeline completo:
    1. Busca dados da API Java.
    2. Treina o modelo de recomendação.
    3. Gera e envia recomendações por e-mail para cada usuário.
    """
    print("=" * 60)
    print("INICIANDO PIPELINE DE RECOMENDAÇÃO DE CACHAÇAS")
    print(
        f"Data e Hora: {pd.Timestamp.now(tz='America/Sao_Paulo').strftime('%d/%m/%Y %H:%M:%S')}"
    )
    print("=" * 60)

    # --- PASSO 1: BUSCAR DADOS FRESCOS DA API ---
    print("\n[PASSO 1/3] Buscando dados da API Java...")
    avaliacoes_df = data_fetcher.get_data_from_api("/avaliacoes")
    cachacas_df = data_fetcher.get_data_from_api("/cachacas")

    if avaliacoes_df is None or cachacas_df is None or avaliacoes_df.empty:
        print(
            "Pipeline abortado: não foi possível buscar dados ou não há avaliações para processar."
        )
        return

    # --- PASSO 2: TREINAR O MODELO ---
    print("\n[PASSO 2/3] Treinando o modelo de IA com os novos dados...")
    model_trainer.train_and_save_model(avaliacoes_df, cachacas_df)

    # --- PASSO 3: GERAR E ENVIAR RECOMENDAÇÕES ---
    print("\n[PASSO 3/3] Gerando e enviando recomendações por e-mail...")
    recommender_system = Recommender()

    if not recommender_system.model:
        print("Pipeline abortado pois o modelo de recomendação não foi carregado.")
        return

    # Pega a lista de usuários únicos que fizeram avaliações
    unique_users = avaliacoes_df["user.id"].unique()
    print(f"Encontrados {len(unique_users)} usuários únicos para processar.")

    for user_id in unique_users:
        print(f"\n--- Processando recomendações para o usuário ID: {user_id} ---")

        # Gera as recomendações para o usuário atual
        recommendations = recommender_system.generate_recommendations(
            user_id=user_id,
            user_ratings_df=avaliacoes_df,
            top_n=3,  # Recomendar o Top 3
        )

        if recommendations:
            # Em um cenário real, você teria um endpoint para buscar o email do usuário.
            # Ex: user_email = data_fetcher.get_user_details(user_id)['email']
            user_email = f"user_{user_id}@exemplo.com"  # << SUBSTITUIR PELA LÓGICA REAL
            print(
                f"Encontradas {len(recommendations)} recomendações. Enviando e-mail para {user_email}."
            )

            email_sender.send_recommendation_email(user_email, recommendations)
        else:
            print(f"Nenhuma nova recomendação encontrada para o usuário {user_id}.")

    print("\n" + "=" * 60)
    print("PIPELINE DE RECOMENDAÇÃO CONCLUÍDO COM SUCESSO!")
    print("=" * 60)


if __name__ == "__main__":
    # Carrega as variáveis de ambiente do arquivo .env para o sistema
    load_dotenv()
    run_recommendation_pipeline()
