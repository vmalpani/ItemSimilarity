from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords as sw
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import sent_tokenize
from scipy import sparse, io

import json
import string
import pickle
import numpy as np


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


def tokenize(sentence):
    stopwords  = set(sw.words('english'))
    punct      = set(string.punctuation)
    stemmer = PorterStemmer()
    tokens = []
    # Break the sentence into tokens
    for token in wordpunct_tokenize(sentence):
        # Apply preprocessing to the token
        token = token.lower()
        token = token.strip()
        token = token.strip('_')
        token = token.strip('*')

        # If stopword, ignore token and continue
        if token in stopwords:
            continue

        # If punctuation, ignore token and continue
        if all(char in punct for char in token):
            continue

        # Stem or lemmatize the token if needed
        # lemma = self.lemmatize(token, tag)
        tokens.append(token)
    return tokens


def generate_data(id_class_description_mapping):
    text = []
    class_name = []
    item_id = []
    for k, v in id_class_description_mapping.items():
        text.append(v['description'])
        class_name.append(v['class'])
        item_id.append(k)
    return text, class_name, item_id


def tfidf_model(data):
    tfidf = TfidfVectorizer(tokenizer=tokenize,stop_words='english')
    model = tfidf.fit(data)
    return model


def vectorize_data(data, model):
    tfs = model.transform(data)
    return tfs


def top_similar_items(query_id, tfidf_matrix, tfidf_model, id_class_description_mapping, item_ids, result_size=5):
    query_class = id_class_description_mapping[query_id]['class']
    item_idx = item_ids.index(query_id)
    query_vector = tfidf_matrix[item_idx]
    item_ids = np.array(item_ids)
    
    similarity_values = cosine_similarity(query_vector, tfidf_matrix)[0]
    top_k = similarity_values.argsort()[::-1]
    count = 0
    result = []
    for item in top_k:
        if count == result_size:
            break
        if item_ids[item] == query_id:
            continue
        if id_class_description_mapping[item_ids[item]]['class'] == query_class:
            result.append(item)
            count += 1
    return item_ids[result], similarity_values[result]


def cluster_similar_items(tfidf_matrix, item_ids, threshold=0.4):
    visited = np.array([0] * len(item_ids))
    item_ids = np.array(item_ids)
    clusters = []
    for n, i in enumerate(visited):
        if i:
            continue
        if n % 10000 == 0:
            print "completed %d" %n
        item = item_ids[n]
        query_vector = tfidf_matrix[n]
        similarity_values = cosine_similarity(query_vector, tfidf_matrix)[0]
        top_k = similarity_values.argsort()[::-1]
        result = [n]
        for idx in top_k:
            if item_ids[idx] == item or visited[idx]:
                continue
            if similarity_values[idx] > threshold:
                result.append(idx)
            else:
                break
        similar_items = item_ids[result]
        visited[result] = 1
        clusters.append(similar_items)
    return clusters


if __name__ == "__main__":

    print "Loading data from disk...\n"
    with open('data/id_class_description_mapping.json') as fp:
        id_class_description_mapping = json.load(fp)

    """
    # uncomment to regenerate the data and tfidf matrix
    print "Generateing label to data mapping...\n"
    X, y_class, y_item_ids = generate_data(id_class_description_mapping)
    with open('data/X', 'wb') as f:
            pickle.dump(X, f)
    with open('data/y_class', 'wb') as f:
            pickle.dump(y_class, f)
    with open('data/y_item_ids', 'wb') as f:
            pickle.dump(y_item_ids, f)

    print "Fitting tfidf model with some preprocessing...\n"
    tfidf_model = tfidf_model(X)

    print "Vectorizing the given data...\n"
    tfidf_matrix = vectorize_data(X, tfidf_model)
    save_sparse_csr("data/tfidf_matrix.npz", tfidf_matrix)
    """

    with open('data/X', 'rb') as f:
            X = pickle.load(f)
    with open('data/y_class', 'rb') as f:
        y_class = pickle.load(f)
    with open('data/y_item_ids', 'rb') as f:
        y_item_ids = pickle.load(f)

    print "Loading tfidf matrix from disk...\n"
    tfidf_matrix = load_sparse_csr("data/tfidf_matrix.npz")

    
    # Q1
    item_id = y_item_ids[0]
    print "Finding the top_k similar items for:\n%s: %s\n" % (item_id, id_class_description_mapping[item_id]['description'])
    similar_items, scores = top_similar_items(item_id, tfidf_matrix, tfidf_model, id_class_description_mapping, y_item_ids, 10)

    for item_id, score in zip(similar_items, scores):
        print "Item id: %s Score:%f Description: %s " % (item_id, score, id_class_description_mapping[item_id]['description'])
    """
    # Q2
    print "Clustering similar item titles...\n"
    clusters = cluster_similar_items(tfidf_matrix, y_item_ids)
    """