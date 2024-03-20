# Import des modules et des packages nécessaires
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import requests
from urllib.parse import urlparse
import asyncio
import json
import pickle
import nltk
import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import euclidean_distances
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from scipy.spatial.distance import cosine

# from sentence_transformers import SentenceTransformer

# Import des constantes
import constants as const  # Assurez-vous que constants.py est présent dans votre répertoire
import warnings

# Ignore UserWarning about token_pattern in CountVectorizer
warnings.filterwarnings("ignore", message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None'")

# Assurez-vous que les ressources nécessaires sont téléchargées
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
LETTER_TEXT = "texte_de_la_lettre"

# Déclaration des variables globales pour stocker les URLs réussies et échouées
success_urls = {}
error_urls = []
error_details = []

query = 'apple'

def tokenize(text):
    """Tokenize the text

    Parameters
    ----------
    text: String
        The message to be tokenized

    Returns
    -------
    List
        List with the clean tokens
    """
    text = text.lower()
    text = re.sub("[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words(const.ENGLISH)]

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    clean_tokens_list = []
    for tok in tokens:
        lemmatizer_tok = lemmatizer.lemmatize(tok).strip()
        clean_tok = stemmer.stem(lemmatizer_tok)
        clean_tokens_list.append(clean_tok)

    return clean_tokens_list

def build_model():
    """Build the model

    Returns
    -------
    sklearn.pipeline.Pipeline
        The model
    """
    pipeline = Pipeline([
        (const.FEATURES, FeatureUnion([

            (const.TEXT_PIPELINE, Pipeline([
                (const.VECT, CountVectorizer(tokenizer=tokenize)),
                (const.TFIDF, TfidfTransformer())
            ]))
    ]))])

    return pipeline

# Fonction pour vérifier si une URL est valide
def is_valid_url(url):
    parsed_url = urlparse(url)
    return parsed_url.scheme in ['http', 'https']

# Fonction pour le scraping des URLs des résultats de recherche Google
async def scrape_google_results_urls(query, browser, max_urls=100):
    page = await browser.new_page()
    await page.goto(f"https://www.google.com/search?q={query}")
    await page.wait_for_load_state("networkidle")

    search_results = await page.query_selector_all('a[href^="http"]')
    urls = [await link.get_attribute("href") for link in search_results]
    valid_urls = [url for url in urls if is_valid_url(url) and "watch" not in url]  # Exclure les liens contenant "watch"

    return valid_urls[:max_urls]

# Fonction pour le scraping avec BeautifulSoup
async def scrape_with_bs(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all("p")
        paragraph_texts = [p.get_text() for p in paragraphs]
        similar_documents = soup.find_all("div", class_="similar-document")
        similar_documents_texts = [doc.get_text() for doc in similar_documents]
        content = {
            "paragraphs": paragraph_texts,
            "similar_documents": similar_documents_texts
        }
        return content
    else:
        return []

# Fonction pour le scraping avec Playwright
async def scrape_with_playwright(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        async with browser:
            try:
                page = await browser.new_page()
                await page.goto(url)
                await page.wait_for_load_state("networkidle", timeout=200000)
                paragraph_elements = await page.query_selector_all("p")
                paragraphs = [await el.inner_text() for el in paragraph_elements]
                similar_documents_elements = await page.query_selector_all(".similar-document")
                similar_documents_texts = [await el.inner_text() for el in similar_documents_elements]
                content = {
                    "paragraphs": paragraphs,
                    "similar_documents": similar_documents_texts
                }
                return content
            except Exception as e:
                error_urls.append(url)
                error_details.append(str(e))  # Enregistrez les détails de l'erreur
                print(f"Error while scraping {url}: {str(e)}")
                return []

def write_to_json(data, filename, explanation=None):
    with open(filename, "w", encoding="utf-8") as json_file:
        if explanation:
            data['explanation'] = explanation
        json.dump(data, json_file, indent=4, ensure_ascii=False)

# search_and_scrape function remains unchanged

# Fonction pour le scraping des URLs et l'exécution du scraping de contenu
async def search_and_scrape(query):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        async with browser:
            google_urls = await scrape_google_results_urls(query, browser)
            print(len(google_urls))
            for url in google_urls:
                scraping_method = scrape_with_bs if is_valid_url(url) else scrape_with_playwright
                data = await scraping_method(url)
                if data:
                    success_urls[url] = data
                    print("Le lien est ouvert :", url)
                else:
                    print("Le lien a recontre une erreur :", url)
            return success_urls

def compare_articles(articles_texts):
    # TF-IDF (déjà présent)
    tfidf_pipeline = build_model()
    tfidf_matrix = tfidf_pipeline.fit_transform(articles_texts)
    tfidf_distances = euclidean_distances(tfidf_matrix, tfidf_matrix)
    
    # Tokenisation pour Word2Vec et Doc2Vec
    tokenized_texts = [tokenize(text) for text in articles_texts]  # Utilisez votre fonction tokenize existante

    # Doc2Vec
    tagged_data = [TaggedDocument(words=text, tags=[str(i)]) for i, text in enumerate(tokenized_texts)]
    doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=20)
    doc2vec_model.build_vocab(tagged_data)
    doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
    doc2vec_distances = [[1 - cosine(doc2vec_model.dv[str(i)], doc2vec_model.dv[str(j)]) if i != j else 0 for j in range(len(articles_texts))] for i in range(len(articles_texts))]

    # Word2Vec (exemple simplifié, ajustez selon besoin)
    word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
    # Pour Word2Vec, la comparaison entre documents nécessite une étape supplémentaire pour convertir les mots en vecteurs documentaires. Ceci est omis pour la simplicité.

    # Retourner plusieurs types de distances
    return {"tfidf": tfidf_distances, "doc2vec": doc2vec_distances}


def prepare_output(distances, articles_data_dict):
    # Liste des identifiants des articles
    articles_identifiers = list(articles_data_dict.keys())

    # Préparation des comparaisons pour TF-IDF
    tfidf_comparisons = prepare_comparisons(distances["tfidf"], articles_identifiers, "TF-IDF")
    
    # Préparation des comparaisons pour Doc2Vec
    doc2vec_comparisons = prepare_comparisons(distances["doc2vec"], articles_identifiers, "Doc2Vec")

    explanation = (
        "This data includes comparisons of document similarities calculated using different methods. "
        "A lower value indicates higher similarity, while a higher value indicates lower similarity."
    )

    output = {
        "tfidf_comparisons": tfidf_comparisons,
        "doc2vec_comparisons": doc2vec_comparisons,
        "explanation": explanation
    }

    return output

def prepare_comparisons(distance_matrix, identifiers, method):
    """Prepare comparisons for a given distance matrix and method."""
    # Si distance_matrix est déjà une liste, pas besoin de convertir
    distances_list = distance_matrix if isinstance(distance_matrix, list) else distance_matrix.tolist()
    comparisons = []
    for i in range(len(distances_list)):
        for j in range(i + 1, len(distances_list)):  # Comparer uniquement les paires uniques sans répétition
            comparisons.append({
                "method": method,
                "document1": identifiers[i],
                "document2": identifiers[j],
                "distance": distances_list[i][j]
            })
    return comparisons

def main():
    articles_data_dict = asyncio.run(search_and_scrape(query))
    articles_texts = [ ' '.join(data['paragraphs'] + data['similar_documents']) for url, data in articles_data_dict.items() ]
    
    if articles_texts:
        tfidf_distances = compare_articles(articles_texts)
        output = prepare_output(tfidf_distances, articles_data_dict)  # Pass both arguments here
        write_to_json(output, "meth_doc_sim_distances_explanation.json")
        print("Les comparaisons des articles ont été enregistrées avec succès, avec des explications.")

if __name__ == "__main__":
    main()
