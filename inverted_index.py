import pandas as pd
import numpy as np
import time
data = pd.read_csv("7allV03.csv")
data.info()

data.head()
data["category"].unique()

from sklearn.utils import shuffle
data = shuffle(data)
data.head()

import string
from nltk.corpus import stopwords

def text_preprocess(text):
    nltk_turkish_stopwords = stopwords.words('turkish')
    # remove punctuations
    trans = str.maketrans('', '', string.punctuation)
    text = text.translate(trans)
    # lowercase the text
    text = text.lower()
    # remove stopwords
    cleaned_text = ""
    for word in text.split():
        if word not in nltk_turkish_stopwords:
            cleaned_text += word + " " 
    return cleaned_text

data["text"] = data["text"].apply(text_preprocess)
data.head()

def retrieve_vectors(data):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer().fit_transform(data["text"])
    docvectors = tfidf.toarray()
    return tfidf, docvectors
    
tfidf, docvectors = retrieve_vectors(data)
docvectors.shape

def run_baseline_test(tfidf):
    start = time.time()
    pairwise_similarity = tfidf * tfidf.T
    end = time.time()

    print("time:", end-start)
    pairwise_similarity = pairwise_similarity.toarray()
    pd.DataFrame(pairwise_similarity)

run_baseline_test(tfidf)

from sklearn.metrics.pairwise import cosine_similarity
def find_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
import matplotlib.pyplot as plt

def plot_results(data_sizes, run_times, comparisons):
        
    plt.figure(figsize=(9, 3))
    plt.subplots_adjust(left=-0.2)
    
    plt.clf()
    plt.subplot(121)
    plt.plot(data_sizes, run_times)
    plt.title('Data Size vs. Runtime')
    plt.xlabel("Data Size")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True)

    plt.subplot(122)
    plt.plot(data_sizes, comparisons)
    plt.title('Data Size vs. Num. of Comparisons')
    plt.xlabel("Data Size")
    plt.ylabel("Num. of Comparisons")
    plt.grid(True)

def run_brute_force_test(data_sizes, docvectors):
    brute_force_run_times = []
    brute_force_comparisons = []
    for data_size in data_sizes:
        test_vectors = docvectors[:data_size]
        num_of_comparisons = 0
        pairwise_similarity = []
        
        start = time.time()
        
        for vector_1 in test_vectors[:len(test_vectors) // 2]:
            cur_vector_sims = []
            for vector_2 in test_vectors:
                sim = find_similarity(vector_1, vector_2)
                cur_vector_sims.append(sim)
                num_of_comparisons += 1
            pairwise_similarity.append(cur_vector_sims)

        end = time.time()
        
        time_passed = end-start
        print("data size:", data_size)
        print("time:", time_passed)
        print("number of comparisons:", num_of_comparisons)
        print()

        brute_force_run_times.append(time_passed)
        brute_force_comparisons.append(num_of_comparisons)

    return brute_force_run_times, brute_force_comparisons, pairwise_similarity
        
data_sizes = list(range(100, 1001, 100))
bf_run_times_1, bf_comparisons_1, pairwise_similarity = run_brute_force_test(data_sizes, docvectors)

pd.DataFrame(pairwise_similarity)
def generate_inverted_index(data: list):
    inv_idx_dict = {}
    for index, doc_text in enumerate(data):
        for word in doc_text.split():
            if word not in inv_idx_dict.keys():
                inv_idx_dict[word] = [index]
            elif index not in inv_idx_dict[word]:
                inv_idx_dict[word].append(index)
    return inv_idx_dict
def run_inverted_index_test(data_sizes, data, docvectors):
    data_sizes = list(range(100, 1001, 100))

    inv_idx_run_times = []
    inv_idx_comparisons = []
    for data_size in data_sizes:
        test_vectors = docvectors[:data_size]
        test_data = data["text"].iloc[:data_size].tolist()
        num_of_comparisons = 0
        pairwise_similarity = []
        inv_idx_dict = generate_inverted_index(test_data)
        
        start = time.time()    

        for cur_doc_index, doc in enumerate(test_data):
            to_compare_indexes = [] 
            # find all the document indexes that have a common word with the current doc
            for word in doc.split():
                to_compare_indexes.extend(inv_idx_dict[word])

            # eliminate duplicates
            to_compare_indexes = list(set(to_compare_indexes))

            # calculate the similarity onlf if the id is larger than 
            # the current document id for better efficiency
            cur_doc_sims = []
            for compare_doc_index in to_compare_indexes:
                if compare_doc_index < cur_doc_index:
                    continue
                sim = find_similarity(test_vectors[cur_doc_index], test_vectors[compare_doc_index])
                num_of_comparisons += 1
                cur_doc_sims.append([compare_doc_index, sim])
            pairwise_similarity.append(cur_doc_sims)

        end = time.time()
        
        time_passed = end-start
        print("data size:", data_size)
        print("time:", time_passed)
        print("number of comparisons:", num_of_comparisons)
        print()

        inv_idx_run_times.append(time_passed)
        inv_idx_comparisons.append(num_of_comparisons)
        
    return inv_idx_run_times, inv_idx_comparisons, pairwise_similarity
        
data_sizes = list(range(100, 1001, 100))
ii_run_times_1, ii_comparisons_1, pairwise_similarity = run_inverted_index_test(data_sizes, data, docvectors)


pd.DataFrame(pairwise_similarity)

plot_results(data_sizes, ii_run_times_1, ii_comparisons_1)



def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results
# ref: https://kavita-ganesan.com/extracting-keywords-from-text-tfidf/#.YCrDUlMzaV4

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

   
def simplify_data(data):
    nltk_turkish_stopwords = stopwords.words('turkish')
    cv = CountVectorizer(max_df=0.85)
    word_count_vector = cv.fit_transform(data["text"].tolist())
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    
    def tfidf_top50(text):
        feature_names=cv.get_feature_names()
        tf_idf_vector = tfidf_transformer.transform(cv.transform([text]))
        sorted_items = sort_coo(tf_idf_vector.tocoo())
        keywords = extract_topn_from_vector(feature_names,sorted_items, 50)
        new_text = ""
        for keyword in keywords:
            new_text += keyword + " "
        return new_text
    
    data["text"] = data["text"].apply(tfidf_top50)
    return data
# shorten the data to save some time
shorter_data = data.iloc[:1001]
simplified_data = simplify_data(shorter_data)
