from flask import Flask, request, jsonify, render_template
from gpt4_summary import gpt4_summary
from scrapper import search_and_scrape, getArticles
from compare_meth import compare_articles, prepare_output
import json
import os
import asyncio
from docx import Document

def write_text_to_word(file_path, text):
    """
    Écrit le texte dans un fichier Word.

    :param file_path: Chemin vers le fichier Word à créer.
    :param text: Texte à écrire dans le fichier.
    """
    try:
        document = Document()
        document.add_paragraph(text)
        document.save(file_path)
        print("Le texte a été écrit avec succès dans le fichier Word.")
    except Exception as e:
        print(f"Erreur lors de l'écriture du fichier Word : {e}")

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
    
def write_to_json(data, filename, explanation=None):
    with open(filename, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

def load_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as json_file:
            return json.load(json_file)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

data_form = None  # Initialisation de la variable globale
openai_api_key =   # Remplacez par votre clé API OpenAI

def main():
    app = Flask(__name__)

    def afficher_donnees_envoyees(donnees):
        # Afficher les valeurs binaires des données envoyées depuis le formulaire HTML
        print("Valeurs binaires des données reçues depuis le formulaire HTML :")
        return donnees

    @app.route('/submit', methods=['POST'])
    def traiter_formulaire():
        global data_form  # Utilisation de la variable globale
        if request.method == 'POST':
            donnees_formulaire = request.json  # Récupérer les données JSON envoyées depuis le formulaire HTML
            data_form = afficher_donnees_envoyees(donnees_formulaire)  # Stocker les données dans la variable globale
            print(data_form)
            # Répondre au client avec un message JSON
            return jsonify({'message': 'Données reçues avec succès !'})

    @app.route('/')
    def afficher_formulaire():
        return render_template('form.html')  # Afficher le formulaire HTML

    @app.after_request
    def afficher_data(response):
        if data_form is not None:
            print(data_form)  # Afficher la variable globale une fois que le formulaire a été soumis et traité
            choose_pdf = []
            if "manuel" in data_form:
                choose_pdf.append(0)
            if "summary" in data_form:
                choose_pdf.append(1)
            if "tutorial" in data_form:
                choose_pdf.append(2)
            print(choose_pdf)
            query = "comment augmenter sa cote en tant qu'artiste ?"
            articles_data_dict1 = asyncio.run(search_and_scrape(query))
            articles_data_dict = getArticles(query)

            # Créer un dictionnaire pour stocker les données des articles au format JSON
            articles_json_dict = {}
            for url, data in articles_data_dict1.items():
                # Créer un dictionnaire pour chaque article
                article_json = {
                        "text": data["paragraphs"],
                }
                # Ajouter l'article au dictionnaire des articles au format JSON
                articles_json_dict[url] = article_json

            # Écrire les données des articles dans le fichier articles_text.json
            with open("articles_text.json", "w", encoding="utf-8") as json_file:
                json.dump(articles_json_dict, json_file, indent=4, ensure_ascii=False)

            # Concaténer tous les paragraphes et documents similaires en une seule chaîne de caractères pour chaque article
            articles_texts = [' '.join(article["text"]) for article in articles_json_dict.values()]

            # Comparer les articles et préparer les données pour le fichier meth_doc_sim_distances_explanation.json
            tfidf_distances = compare_articles(articles_texts)
            output = prepare_output(tfidf_distances, articles_data_dict1)
            write_to_json(output, "meth_doc_sim_distances_explanation.json")

            print("Les comparaisons des articles ont été enregistrées avec succès, avec des explications.")
            
            tfidf_data = load_json_file("meth_doc_sim_distances_explanation.json")
            article_data = load_json_file("articles_text.json")
            global_article_texts = []  # Renamed from global_summary_parts to reflect the actual content

            if tfidf_data and article_data:
                top_links_tfidf = sorted(tfidf_data["tfidf_comparisons"], key=lambda x: x["distance"], reverse=True)[:5]

                for link in top_links_tfidf:
                    article_url = link["document2"]
                    if article_url in article_data:
                        # Retrieve the full text instead of summarizing
                        article_text = "\n".join(article_data[article_url]["text"])
                        global_article_texts.append(article_text)  # Append the full text
                    else:
                        print(f"Impossible de récupérer le texte de l'article pour l'URL : {article_url}")

            # Optionally, you can still write the full texts to a JSON file if needed
            write_to_json(global_article_texts, "global_article_texts.json")

            file_path = "global_article_texts.json"  # Assurez-vous que le chemin vers le fichier est correct
            output_file_path = "summary.txt"  # Chemin vers le nouveau fichier de résumé
            text = read_text_from_file(file_path)
            if not text.startswith("Erreur lors de la lecture du fichier"):
                response0, response1, response2 = gpt4_summary(text, openai_api_key, choose_pdf)
                sujet = str(response0).split('\n')[0]
                print("=====================")
                print(response0)
                print("=====================")
                print(response1)
                print("=====================")
                print(response2)
                print("=====================")
                # Obtenir le chemin absolu du répertoire de travail actuel
                chemin_dossier = os.getcwd()
                # print(chemin_dossier)
                # Créer un dossier avec le nom du sujet s'il n'existe pas déjà
                dossier = sujet.replace(" ", "_")  # Remplacer les espaces par des underscores pour éviter les problèmes de chemin
                chemin_complet_dossier = os.path.join(chemin_dossier, dossier)
                # if not os.path.exists(chemin_complet_dossier):
                os.makedirs(chemin_complet_dossier)
                # Écrire les fichiers dans le dossier créé
                print(data_form)
                if 0 in choose_pdf:
                    write_text_to_word(os.path.join(dossier, f"Intro_{sujet}.docx"), response0)
                if 1 in choose_pdf:
                    write_text_to_word(os.path.join(dossier, f"User_Manual_{sujet}.docx"), response1)
                if 2 in choose_pdf:
                    write_text_to_word(os.path.join(dossier, f"Sommaire_{sujet}.docx"), response2)
                    # write_text_to_word(os.path.join(dossier, f"1Part_{sujet}.docx"), response3)
                    # write_text_to_word(os.path.join(dossier, f"2Part_{sujet}.docx"), response4)
                    # write_text_to_word(os.path.join(dossier, f"3Part_{sujet}.docx"), response5)
                    # write_text_to_word(os.path.join(dossier, f"4Part_{sujet}.docx"), response6)
                    # write_text_to_word(os.path.join(dossier, f"5Part_{sujet}.docx"), response7)
            else:
                print(text)
        return response

    if __name__ == '__main__':
        app.run(debug=True)

if __name__ == "__main__":
    main()

