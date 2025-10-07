# email_sender.py
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List


def _format_html_email(recommendations: List[Dict[str, Any]]) -> str:
    """
    Cria o corpo do e-mail em HTML a partir da lista de recomendações.
    """
    sender_name = os.getenv("EMAIL_SENDER_NAME", "Pingou")

    # Início do HTML com estilo CSS incorporado
    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; line-height: 1.6; background-color: #f9f9f9; color: #333; }}
            .container {{ max-width: 600px; margin: 20px auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background-color: #ffffff; }}
            .header {{ font-size: 24px; font-weight: bold; color: #8B4513; text-align: center; border-bottom: 2px solid #8B4513; padding-bottom: 10px; margin-bottom: 20px; }}
            .recommendation {{ margin-bottom: 20px; padding: 15px; border: 1px solid #eee; border-radius: 5px; background-color: #fafafa; }}
            .rec-title {{ font-weight: bold; font-size: 18px; color: #BF5700; }}
            .rec-details {{ font-size: 14px; color: #555; margin-top: 5px; }}
            .rec-description {{ font-style: italic; color: #666; margin-top: 8px; }}
            .footer {{ text-align: center; font-size: 12px; color: #999; margin-top: 30px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">Recomendações Especiais para Você!</div>
            <p>Olá, apreciador(a) de cachaça!</p>
            <p>Com base nas suas últimas avaliações, nosso sistema de recomendação selecionou algumas jóias que têm tudo a ver com o seu paladar:</p>
    """

    # Adiciona cada recomendação dinamicamente
    for rec in recommendations:
        html += f"""
        <div class="recommendation">
            <div class="rec-title">{rec.get('nome', 'Nome Indisponível')}</div>
            <div class="rec-details">
                <b>Tipo:</b> {rec.get('tipoCachaca', 'N/A')} | <b>Região:</b> {rec.get('regiao', 'N/A')}
            </div>
            <div class="rec-description">
                {rec.get('descricao', 'Descrição não disponível.')}
            </div>
        </div>
        """

    # Fim do HTML
    html += f"""
            <p>Esperamos que goste das sugestões!</p>
            <div class="footer">
                Atenciosamente,<br>
                <b>Equipe {sender_name}</b>
            </div>
        </div>
    </body>
    </html>
    """
    return html


def send_recommendation_email(
    recipient_email: str, recommendations: List[Dict[str, Any]]
):
    """
    Envia um e-mail com as cachaças recomendadas para um destinatário.
    """
    if not recommendations:
        print(f"Nenhuma recomendação para enviar para {recipient_email}.")
        return

    # Busca as credenciais de e-mail do arquivo .env
    host = os.getenv("EMAIL_HOST")
    port = int(os.getenv("EMAIL_PORT", 587))
    user = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASSWORD")
    sender_email = user
    sender_name = os.getenv("EMAIL_SENDER_NAME", "Pingou")

    # Verifica se as credenciais estão presentes
    if not all([host, port, user, password]):
        print(
            "ERRO: As variáveis de ambiente do e-mail não estão configuradas no arquivo .env."
        )
        return

    # Montando a mensagem de e-mail
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Suas recomendações de cachaça da semana | {sender_name}"
    msg["From"] = f"{sender_name} <{sender_email}>"
    msg["To"] = recipient_email

    html_body = _format_html_email(recommendations)
    msg.attach(MIMEText("Temos novas recomendações de cachaça para você.", "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        print(f"Tentando enviar e-mail para {recipient_email}...")
        with smtplib.SMTP(host, port) as server:
            server.starttls()  # Ativa a segurança
            server.login(user, password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        print("E-mail enviado com sucesso!")
    except Exception as e:
        print(f"FALHA ao enviar e-mail para {recipient_email}: {e}")
