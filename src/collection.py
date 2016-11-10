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


def _tokenize(sentence):
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


class Collection:
    """Base class for Collections"""

    def __init__(self, name):
        self.name = name
        self.ids, self.data = self._parse_file()
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
        # FIXME: we open the file in append mode to ensure that we can write
        # outlier clusters to the same file. Make sure you delete the file
        # before a fresh run.
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

    def _generate_tfidf_matrix(self):
        """Load precomputed tfidf matrix if present or compute a new one"""
        if os.path.exists("data/tfidf_matrix_%s.npz" % self.name):
            tfidf_matrix = helper.load_sparse_csr("data/tfidf_matrix_%s.npz"
                                                  % self.name)
        else:
            tfidf = TfidfVectorizer(tokenizer=_tokenize,
                                    stop_words='english')
            tfidf.fit(self.data)
            tfidf_matrix = tfidf.transform(self.data)
            helper.save_sparse_csr("data/tfidf_matrix_%s.npz" % self.name,
                                   tfidf_matrix)
        return tfidf_matrix

    def _run_db_scan(self, idxs, thresh=0.4, n_samples=5):
        """Run db scan on subsample idxs"""
        subset_tfidf = self.tfidf_matrix[idxs]
        dist = 1 - cosine_similarity(subset_tfidf)

        # step 1: use density clustering
        clust = DBSCAN(eps=thresh, min_samples=n_samples, metric="precomputed")
        clust.fit(dist)
        return clust.labels_

    def _apprx_nn_search(self, input_vecs, data_to_return, predict_vecs,
                         result_size=10, num_clusters=5, get_distance=False):
        """
        Builds a search index from the input vectors and uses in for fast
        nn lookup of prediction vectors
        input_vecs: sparse tfidf vectors of input data
        data_to_return: data to be returned corresponding to the nn returned
                        ex. we use it to return cluster number or item_id
        predict_vecs: sparse tfidf vectors of nn query data
        result_size: number of results to be returned
        num_clusters: number of clusters to be searched at each level
        get_distance: if true, returns the distance of the nearest neighbors
        """
        cp = snn.ClusterIndex(input_vecs, data_to_return)
        result = cp.search(predict_vecs, k=result_size+1,
                           k_clusters=num_clusters,
                           return_distance=get_distance)
        return result

    def top_similar_items(self, result_size=10):
        """
        Generates top k similar items for each item in the whole collection
        using pysparnn which implements clustering based approximate nn lookup
        """
        result = self._apprx_nn_search(self.tfidf_matrix,
                                       zip(self.ids, self.data),
                                       self.tfidf_matrix, result_size)
        self._write_results(result)

    def generate_clusters(self):
        """
        Clusters similar items using db scan on tfidf vectors of a subset of
        the collection and does assigns the left out items to clusters via
        approximate nearest neighbor search
        """
        # shuffle idxs over all the items in the file
        perm_idxs = range(self.tfidf_matrix.shape[0])
        random.shuffle(perm_idxs)

        # sample 20k idxs to generate clusters
        first_idxs = perm_idxs[:20000]
        rest_idxs = perm_idxs[20000:]

        # create a distance metric from cosine similarity
        first_preds = self._run_db_scan(first_idxs, 0.2, 5)

        all_idxs = first_idxs
        all_preds = first_preds

        # if we have some items left after dbscan
        if rest_idxs and len(rest_idxs) > 0:
            # get rid of noisy outliers
            pruned_pred = []
            pruned_idx = []
            for _idx, _pred in zip(first_idxs, first_preds):
                if _pred >= 0:
                    pruned_pred.append(_pred)
                    pruned_idx.append(_idx)

            result = self._apprx_nn_search(self.tfidf_matrix[pruned_idx],
                                           pruned_pred,
                                           self.tfidf_matrix[rest_idxs],
                                           get_distance=True)

            # step 2: assign each example a cluster by max voting
            p = multiprocessing.Pool(multiprocessing.cpu_count())
            rest_preds = p.map(_max_vote, result)
            p.close()
            p.join()

            # merge back results from step 1 and step 2
            all_idxs += rest_idxs
            all_preds = np.hstack((first_preds, rest_preds))

            # print zip(rest_preds, np.array(self.data)[rest_idxs])[:20]

        assert len(all_idxs) == all_preds.shape[0]

        # last pass to check if we mistook a cluster altogether as outlier
        # as none of its members were choosen in the initial clustering phase
        outlier_idxs = np.array(perm_idxs)[all_preds == -1]
        if len(outlier_idxs) > 0:
            outliers_preds = self._run_db_scan(outlier_idxs, 0.2, 5)

        # write main pass clusters
        self._write_clusters(all_preds, all_idxs)

        # outliers pass clusters
        self._write_clusters(outliers_preds, outlier_idxs,
                             suffix='Outliers')


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
    """
    item_categories = ['cases', 'cell_phones', 'laptops', 'mp3_players']
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', required=True,
                        help='Enter item category: cases, cell_phones, laptops, mp3_players')  # noqa
    args = parser.parse_args()
    if args.category in item_categories:
        c = Collection(args.category)
        print "Generating top 10 similar items...\n"
        t1 = time.time()
        c.top_similar_items()
        print "Time taken for similarity: %s" % str(time.time()-t1)

        print "Clustering similar items together...\n"
        t2 = time.time()
        c.generate_clusters()
        print "Time taken for clustering: %s" % str(time.time()-t2)
    else:
        print "Incorrect category."
        print "Please choose from cases, cell_phones, laptops, mp3_players."
