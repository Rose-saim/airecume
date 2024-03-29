import importlib.util
import subprocess
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import requests
from urllib.parse import urlparse
import nltk
import warnings
import json
# Déclaration des variables globales pour stocker les URLs réussies et échouées
success_urls = {}
error_urls = []
error_details = []

# Vérifier et installer nltk si nécessaire
def check_library(library_name):
    spec = importlib.util.find_spec(library_name)
    if spec is None:
        print(f"{library_name} is not installed. Installing...")
        try:
            # Installer la bibliothèque
            subprocess.check_call(["pip", "install", library_name])
            print(f"{library_name} installed successfully.")
        except Exception as e:
            print(f"Error installing {library_name}: {e}")
    else:
        print(f"{library_name} is already installed")

# Ignore UserWarning about token_pattern in CountVectorizer
warnings.filterwarnings("ignore", message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None'")

# Assurez-vous que les ressources nécessaires sont téléchargées
if check_library('nltk'):
    nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
LETTER_TEXT = "texte_de_la_lettre"

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

def scrape_with_bs(url):
    # Utilisation d'une session pour les requêtes
    with requests.Session() as session:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }
            response = requests.get(url, headers=headers, timeout=10)

            # Vérifier le statut de la réponse
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all("p")
                paragraph_texts = [p.get_text() for p in paragraphs]
                similar_documents = soup.find_all("div", class_="similar-document")
                similar_documents_texts = [doc.get_text() for doc in similar_documents]
                
                # Créer un dictionnaire pour stocker les données de l'article
                article_data = {
                    "paragraphs": paragraph_texts,
                    "similar_documents": similar_documents_texts
                }
                
                # Retourner le dictionnaire des données de l'article
                return article_data
            else:
                print(f"Erreur : Le serveur a retourné un statut {response.status_code}")
                return {}
        except requests.exceptions.ConnectionError as e:
            # Gérer l'exception de connexion
            print(f"Erreur de connexion : {e}")
            return {}
        except requests.exceptions.Timeout as e:
            # Gérer l'exception de timeout
            print(f"Le délai d'attente est dépassé : {e}")
            return {}
        except requests.exceptions.RequestException as e:
            # Gérer toutes les autres exceptions possibles de requests
            print(f"Erreur lors de la requête : {e}")
            return {}

# Fonction pour le scraping avec Playwright
async def scrape_with_playwright(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        async with browser:
            try:
                page = await browser.new_page()
                await page.goto(url)
                await page.wait_for_load_state("networkidle", timeout=300000)
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

# search_and_scrape function remains unchanged
# Fonction pour le scraping des URLs et l'exécution du scraping de contenu
async def search_and_scrape(query):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        async with browser:
            google_urls = await scrape_google_results_urls(query, browser)
            for url in google_urls:
                scraping_method = scrape_with_bs if is_valid_url(url) else scrape_with_playwright
                article_data = scraping_method(url)  # Remove 'await' keyword here
                if article_data:
                    success_urls[url] = article_data
            return success_urls


def getArticles(query):
    url = "https://google.serper.dev/news"
 
    payload = json.dumps({
        "q": query,
        "gl": "fr",
        "hl": "fr",
        "num": 20
    })
    headers = {
        'X-API-KEY': '58298d6439af0085aaad81ba7b9c89d275a5916d',
        'Content-Type': 'application/json'
    }
 
    response = requests.request("POST", url, headers=headers, data=payload)
    return response