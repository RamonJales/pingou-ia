# pingou-ia

/recomendacao-service
|-- main.py # Script principal que orquestra tudo
|-- data_fetcher.py # Módulo para buscar dados da API Java
|-- pipeline.py # Módulo para treinar e salvar o modelo
|-- recommender.py # Módulo para gerar recomendações com o modelo
|-- email_sender.py # Módulo para enviar os e-mails
|-- requirements.txt # Dependências (pandas, scikit-learn, requests, lightfm)
|-- .env # Arquivo para guardar segredos (API key, credenciais de email)

