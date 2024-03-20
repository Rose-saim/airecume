import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from langdetect import detect

def download_nltk_resources():
    resources = ['punkt', 'stopwords']
    for resource in resources:
        try:
            nltk.download(resource)
        except Exception as e:
            print(f"Error downloading {resource}: {e}")

def load_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as json_file:
            return json.load(json_file)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"  # Default to English if detection fails

def get_stop_words(language):
    try:
        return set(stopwords.words(language))
    except:
        return set(stopwords.words('english'))

def summarize_text(text):
    language = detect_language(text)
    stop_words = get_stop_words(language)
    
    try:
        sentences = sent_tokenize(text, language=language)
    except LookupError:
        nltk.download('punkt')
        sentences = sent_tokenize(text, language='english')

    words = [word.lower() for word in word_tokenize(text) if word.isalnum() and word.lower() not in stop_words]
    freq_dist = FreqDist(words)
    ranked_sentences = [(sent, sum(freq_dist[word] for word in word_tokenize(sent.lower()) if word in freq_dist)) for sent in sentences]
    ranked_sentences.sort(key=lambda x: x[1], reverse=True)

    summary_sentences = [sent for sent, score in ranked_sentences[:3]]
    return ' '.join(summary_sentences)

if __name__ == "__main__":
    download_nltk_resources()
    
    tfidf_data = load_json_file("meth_doc_sim_distances_explanation.json")
    article_data = load_json_file("articles_data.json")
    global_summary_parts = []

    if tfidf_data and article_data:
        top_links_tfidf = sorted(tfidf_data["tfidf_comparisons"], key=lambda x: x["distance"], reverse=True)[:5]

        for link in top_links_tfidf:
            article_url = link["document2"]
            if article_url in article_data:
                article_text = "\n".join(article_data[article_url]["paragraphs"])
                summary = summarize_text(article_text)
                global_summary_parts.append(summary)
            else:
                print(f"Impossible de récupérer le texte de l'article pour l'URL : {article_url}")

    # Combining individual summaries into a global summary
    global_summary = ' '.join(global_summary_parts)
    
    # Saving the global summary to a file
    with open("global_article_summary.json", "w", encoding="utf-8") as summary_file:
        summary_file.write(global_summary)
    print("Le résumé global de tous les articles a été sauvegardé dans 'global_article_summary.txt'.")
