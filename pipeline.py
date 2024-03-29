
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import constants as const
import constants as const  # Assurez-vous que constants.py est présent dans votre répertoire
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

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
    # Supprimer les lignes vides du texte
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    text = '\n'.join(non_empty_lines)
    
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