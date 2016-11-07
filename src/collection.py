from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords as sw
from nltk import wordpunct_tokenize
import pysparnn as snn

import helper

import os
import string
import warnings
warnings.filterwarnings("ignore")


FILE_PATH = 'data/%s.txt'
RESULT_PATH = 'results/%s.txt'


class Collection:
    def __init__(self, name):
        self.tfidf = TfidfVectorizer(tokenizer=self._tokenize,
                                     stop_words='english')
        self.name = name

    def _parse_file(self):
        ids = []
        data = []
        with open(FILE_PATH % self.name) as fp:
            for line in fp:
                tmp = line.rstrip().split('\t')
                assert len(tmp) == 4
                ids.append(int(tmp[0]))
                data.append(tmp[-1])
        return ids, data

    def _write_results(self, ids, data, result):
        with open(RESULT_PATH % (self.name + '_results'), 'w') as fp:
            for item_id, title, similar_items in zip(ids, data, result):
                fp.write('\nItem: %d\t%s\n' % (item_id, title))
                fp.write('Similar Items:\n')
                for i in similar_items[1:]:
                    fp.write('%s\t%s\n' % (i[0], i[1]))

    def _tokenize(self, sentence):
        stopwords = set(sw.words('english'))
        punct = set(string.punctuation)
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

    def _generate_tfidf_matrix(self, data):
        self.tfidf.fit(data)
        tfidf_matrix = self.tfidf.transform(data)
        helper.save_sparse_csr("data/tfidf_matrix_%s.npz" % self.name,
                               tfidf_matrix)
        return tfidf_matrix

    def top_similar_items(self, result_size=10):
        # read the input txt file given
        ids, data = self._parse_file()

        # load a precomputed tfidf matrix if present
        # or fit a new tfidf vectorizer
        if os.path.exists("data/tfidf_matrix_%s.npz" % self.name):
            tfidf_matrix = helper.load_sparse_csr("data/tfidf_matrix_%s.npz"
                                                  % self.name)
        else:
            tfidf_matrix = self._generate_tfidf_matrix(data)

        # build search index using pysparnn
        cp = snn.ClusterIndex(tfidf_matrix, zip(ids, data))

        # nn lookup
        # k: number of results required (one is added to result_size as the
        #    first item in the result is query item itself)
        # k_clusters: number of clusters to search at each level
        result = cp.search(tfidf_matrix, k=result_size+1, k_clusters=5,
                           return_distance=False)

        self._write_results(ids, data, result)


if __name__ == "__main__":
    """
    Item Categories: 'cases', 'cell_phones', 'laptops', 'mp3_players'

    'cases': Time to build search index ~ 10s
             Time for nn lookup ~ 481.17s / 99999 ~ 5ms

    'cell_phones': Time to build search index ~ 15s
                   Time for nn lookup ~ 468.73s / 99999 ~ 5ms

    'laptops': Time to build search index ~ 5s
               Time for nn lookup ~ 235s / 56638 ~ 4ms

    'mp3_players': Time to build search index ~ 2.4s
                   Time for nn lookup ~ 100s / 24691 ~ 4ms
    """
    c = Collection('mp3_players')
    c.top_similar_items()
