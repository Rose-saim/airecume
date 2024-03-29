
from sklearn.metrics.pairwise import euclidean_distances
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from scipy.spatial.distance import cosine
from pipeline import build_model, tokenize

# def detect_language(text):
#     try:
#         return detect(text)
#     except:
#         return "en"  # Fallback to English

# def get_stop_words(language):
#     try:
#         return set(stopwords.words(language))
#     except:
#         return set(stopwords.words('english'))

# def summarize_text(text):
#     language = detect_language(text)
#     stop_words = get_stop_words(language)
    
#     try:
#         sentences = sent_tokenize(text, language=language)
#     except LookupError:
#         nltk.download('punkt')
#         sentences = sent_tokenize(text, language='english')

#     if sentences:
#         sentences.pop(0)  # Remove the first paragraph

#     words = [word.lower() for word in word_tokenize(text) if word.isalnum() and word.lower() not in stop_words]
#     freq_dist = FreqDist(words)
#     ranked_sentences = [(sent, sum(freq_dist[word] for word in word_tokenize(sent.lower()) if word in freq_dist)) for sent in sentences]
#     ranked_sentences.sort(key=lambda x: x[1], reverse=True)

#     summary_sentences = [sent for sent, score in ranked_sentences[:3]]
#     return ' '.join(summary_sentences)

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
            if (distances_list[i][j] > 1):
                comparisons.append({
                    "method": method,
                    "document1": identifiers[i],
                    "document2": identifiers[j],
                    "distance": distances_list[i][j]
                })
    return comparisons

