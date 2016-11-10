from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

from nltk.corpus import stopwords as sw
from nltk import wordpunct_tokenize
import multiprocessing
import pysparnn as snn
import numpy as np

import helper

import time
import random
import argparse
import os
import string
import warnings
warnings.filterwarnings("ignore")


FILE_PATH = 'data/%s.txt'
RESULT_PATH = 'results/%s.txt'
VOTING_THRESHOLD = 0.6


def _max_vote(result):
    """return cluster with max votes having similarity above threshold"""
    # if min neighbor distance too high, consider noise
    if result[0][0] > VOTING_THRESHOLD:
        return -1
    else:
        # gather votes from candidates that are similar enough
        c_labels = [x[1] for x in result if x[0] < VOTING_THRESHOLD]
        if len(c_labels) > 0:
            return max(set(c_labels), key=c_labels.count)
        else:
            return -1


class Collection:
    """Base class for Collections"""

    def __init__(self, name):
        self.name = name
        self.ids, self.data = self._parse_file()
        # self.tfidf = TfidfVectorizer(tokenizer=self._tokenize,
        #                             stop_words='english')
        self.tfidf_matrix = self._generate_tfidf_matrix()

    def _parse_file(self):
        """Parse the input txt file into item ids and description"""
        ids = []
        data = []
        with open(FILE_PATH % self.name) as fp:
            for line in fp:
                tmp = line.rstrip().split('\t')
                assert len(tmp) == 4
                ids.append(int(tmp[0]))
                data.append(tmp[-1])
        return ids, data

    def _write_results(self, result):
        """Write similar items to a file"""
        with open(RESULT_PATH % (self.name + '_results'), 'w') as fp:
            for item_id, title, similar_items in zip(self.ids,
                                                     self.data,
                                                     result):
                fp.write('\nItem: %d\t%s\n' % (item_id, title))
                fp.write('Similar Items:\n')
                for i in similar_items[1:]:
                    fp.write('%s\t%s\n' % (i[0], i[1]))

    def _write_clusters(self, preds, perm_idxs, suffix=''):
        """Write cluster members to a file"""
        clabels = np.unique(preds)
        # for each cluster print out the items that belong to it
        with open(RESULT_PATH % (self.name + '_clusters'), 'a') as fp:
            for i in range(clabels.shape[0]):
                # skip for noisy outliers indicated with cluster number -1
                if clabels[i] < 0:
                    continue

                cmem_ids = np.where(preds == clabels[i])[0]
                fp.write('Cluster#%s#%d\n' % (suffix, i))
                for cmem_id in cmem_ids:
                    fp.write('%d\t%s\n' % (self.ids[perm_idxs[cmem_id]],
                                           self.data[perm_idxs[cmem_id]]))
                fp.write('\n')

    def _tokenize(self, sentence):
        """Tokenizer used by tfidf vectorizer"""
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

    def _generate_tfidf_matrix(self):
        # load a precomputed tfidf matrix if present
        # or fit a new tfidf vectorizer
        if os.path.exists("data/tfidf_matrix_%s.npz" % self.name):
            tfidf_matrix = helper.load_sparse_csr("data/tfidf_matrix_%s.npz"
                                                  % self.name)
        else:
            tfidf = TfidfVectorizer(tokenizer=self._tokenize,
                                    stop_words='english')
            tfidf.fit(self.data)
            tfidf_matrix = tfidf.transform(self.data)
            helper.save_sparse_csr("data/tfidf_matrix_%s.npz" % self.name,
                                   tfidf_matrix)
        return tfidf_matrix

    def _run_db_scan(self, idxs, thresh=0.4, n_samples=5):
        subset_tfidf = self.tfidf_matrix[idxs]
        dist = 1 - cosine_similarity(subset_tfidf)

        # step 1: use density clustering
        print "applying dbscan clustering..."
        t1 = time.time()
        clust = DBSCAN(eps=thresh, min_samples=n_samples, metric="precomputed")
        clust.fit(dist)
        print time.time() - t1
        return clust.labels_

    """
    def apprx_cluster_search(self, input_vecs, data_to_return, predict_vecs,
                             result_size=10, k_clusters=5, get_distance=False):
        # nn lookup
        # k: number of results required (one is added to result_size as the
        #    first item in the result will be the query item itself)
        # k_clusters: number of clusters to search at each level
        cp = snn.ClusterIndex(input_vecs, data_to_return)
        result = cp.search(predict_vecs, k=result_size+1, k_clusters=5,
                           return_distance=False)
        return result
    """

    def top_similar_items(self, result_size=10):
        # build search index using pysparnn
        cp = snn.ClusterIndex(self.tfidf_matrix, zip(self.ids, self.data))

        # nn lookup
        # k: number of results required (one is added to result_size as the
        #    first item in the result will be the query item itself)
        # k_clusters: number of clusters to search at each level
        result = cp.search(self.tfidf_matrix, k=result_size+1, k_clusters=5,
                           return_distance=False)

        self._write_results(result)

    def generate_clusters(self, thresh=0.4, n_samples=5):
        # shuffle idxs over all the items in the file
        perm_idxs = range(self.tfidf_matrix.shape[0])
        random.shuffle(perm_idxs)

        # sample 20k idxs to generate clusters
        first_idxs = perm_idxs[:20000]
        rest_idxs = perm_idxs[20000:21000]

        print "computing distance from cosine similarity..."
        # create a distance metric from cosine similarity
        first_preds = self._run_db_scan(first_idxs)

        all_idxs = first_idxs
        all_preds = first_preds

        # if we have some items left after dbscan
        if rest_idxs and len(rest_idxs) > 0:
            print "building search index from first chunk..."
            # get rid of noisy outliers
            pruned_pred = []
            pruned_idx = []
            for _idx, _pred in zip(first_idxs, first_preds):
                if _pred >= 0:
                    pruned_pred.append(_pred)
                    pruned_idx.append(_idx)
            t2 = time.time()
            # build a search index for fast approximate nn lookup
            cp = snn.ClusterIndex(self.tfidf_matrix[pruned_idx], pruned_pred)
            print time.time() - t2

            # find the k most similar items for each of the remaining items
            print "find k nearest neighbors of each of the remaining items..."
            t3 = time.time()

            # results are returned as list of tuples [(dist, cluster_num), ...]
            result = cp.search(self.tfidf_matrix[rest_idxs], k=10,
                               k_clusters=5, return_distance=True)

            # result = cp.search(self.tfidf_matrix[rest_idxs], k=20,
            #                   k_clusters=15, return_distance=False)
            print time.time() - t3

            t4 = time.time()
            print "label the remaining item by max voting"

            # step 2: assign each example a cluster by max voting
            p = multiprocessing.Pool(multiprocessing.cpu_count())
            rest_preds = p.map(_max_vote, result)
            p.close()
            p.join()

            print time.time() - t4
            # merge back results from step 1 and step 2
            all_idxs += rest_idxs
            all_preds = np.hstack((first_preds, rest_preds))

            # print zip(rest_preds, np.array(self.data)[rest_idxs])[:20]

        assert len(all_idxs) == all_preds.shape[0]

        # last pass to check if we mistook a cluster altogether as outlier
        # as none of its members were choosen in the initial clustering phase
        outlier_idxs = np.array(perm_idxs)[all_preds == -1]
        if len(outlier_idxs) > 0:
            outliers_preds = self._run_db_scan(outlier_idxs)

        # write main pass clusters
        self._write_clusters(all_preds, all_idxs)

        # outliers pass clusters
        self._write_clusters(outliers_preds, outlier_idxs, suffix='Outliers')


if __name__ == "__main__":
    """
    Item Categories: 'cases', 'cell_phones', 'laptops', 'mp3_players'
    All measurements have been done on my macbook: 3GHz i7, 16G Memory
    Q1: Top k similar items analysis

    'cases': Time to build search index ~ 10s
             Time for nn lookup ~ 481.17s / 99999 ~ 5ms

    'cell_phones': Time to build search index ~ 15s
                   Time for nn lookup ~ 468.73s / 99999 ~ 5ms

    'laptops': Time to build search index ~ 5s
               Time for nn lookup ~ 235s / 56638 ~ 4ms

    'mp3_players': Time to build search index ~ 2.4s
                   Time for nn lookup ~ 100s / 24691 ~ 4ms

    Q2: Clustering
    'cases':
              Thresholds: 0.4, 5, 15, 20
              Time for dbscan ~ 2s
              Time to build the search index ~ 0.5s
              Time per nn lookup: 915s / 89999 ~ 10ms

    'cell_phones':
              Thresholds: 0.2, 5, 10, 5, 0.6
              Time for dbscan ~ 3s
              Time to build the search index ~ 1s
              Time per nn lookup: 858s / 89999 ~  9.5ms

    'laptops':
              Time for dbscan ~ 1.6s
              Time to build the search index ~ 0.2s
              Time per nn lookup: 130s / 36638 ~  3ms

    'mp3_players':  Time for dbscan ~ 2s
              Time to build the search index ~ 0.5s
              Time per nn lookup: 10.74s / 1000 ~  10ms
    """
    item_categories = ['cases', 'cell_phones', 'laptops', 'mp3_players']
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', required=True,
                        help='Enter item category: cases, cell_phones, laptops, mp3_players')  # noqa
    args = parser.parse_args()
    if args.category in item_categories:
        c = Collection(args.category)
        print "Generating top 10 similar items...\n"
        # t1 = time.time()
        # c.top_similar_items()
        # print "time take for similarity: %s" % str(time.time()-t1)

        print "Clustering similar items together...\n"
        t2 = time.time()
        c.generate_clusters(0.2, 5)
        print "time take for clustering: %s" % str(time.time()-t2)
    else:
        print "Incorrect category."
        print "Please choose from cases, cell_phones, laptops, mp3_players."
