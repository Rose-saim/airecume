from openai import OpenAI
import os
# from dotenv import load_dotenv

# Charge les variables d'environnement à partir du fichier .env
# load_dotenv()

def read_text_from_file(file_path):
    """
    Lit le contenu d'un fichier et le retourne sous forme de texte.

    :param file_path: Chemin vers le fichier à lire.
    :return: Contenu du fichier en tant que chaîne de caractères.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Erreur lors de la lecture du fichier : {e}"

def gpt4_summary(text, openai_api_key):
    """
    Utilise GPT-4 pour résumer le texte donné.
    :param text: Texte à résumer.
    :param openai_api_key: Clé API pour OpenAI.
    :return: Résumé du texte.
    """

    client =  OpenAI(api_key='')

    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Assurez-vous que "gpt-4" est le bon nom du modèle au moment de l'exécution
            messages=[
                {"role": "system", "content": "Résume ce qui suit :"},
                {"role": "user", "content": text}
            ],
            temperature=0.5,
            max_tokens=150,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return(response.choices[0].message.content)
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

# Exemple d'utilisation
if __name__ == "__main__":
    file_path = "global_article_summary.txt"  # Assurez-vous que le chemin vers le fichier est correct
    openai_api_key = "votre_clé_api_openai_ici"  # Remplacez par votre clé API OpenAI
    text = read_text_from_file(file_path)
    
    if not text.startswith("Erreur lors de la lecture du fichier"):
        summary = gpt4_summary(text, openai_api_key)
        print(summary)
    else:
        print(text)
