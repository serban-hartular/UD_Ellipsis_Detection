import itertools
from typing import List, Dict, Tuple

import sklearn
import pandas as pd
import pickle
import warnings

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

import tree_path as tp

from robert_client import RoBERT_Client
client = RoBERT_Client('http://127.0.0.1:5001', timeout=10)

with open('./ellipticity_detection.p', 'rb') as handle:
    classifiers = pickle.load(handle)

classifier = classifiers['LogisticRegression']['elliptic'][1]

dl = tp.DocList.from_conllu('./cancan21-annot-2.3-vpe.conllu')

for doc in dl:
    for s_index, sentence in enumerate(doc):
        _context = doc[max(0, s_index-1):s_index+2] # previous and next sentences
        _context = [s.projection_nodes() for s in _context]
        context = [] # can't get itertools.chain to work!
        for _nl in _context: context.extend(_nl)
        vectors = None
        node_list = sentence.projection_nodes()
        for node in node_list:
            what_is = node.sdata('misc.Ellipsis')
            if what_is != 'pVPE':
                continue
            if not vectors:
                vectors = client.get_vectors([n.sdata('form') for n in context])
            node_index = context.index(node)
            node_vector = vectors['word_vectors'][node_index]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # pred = classifier.predict([node_vector])[0]
                prob = classifier.predict_proba([node_vector])[0][1]
                pred = int(prob > 0.5)
            context_text = ' '.join(['*%s*' % _n.sdata('form') if _n == node else _n.sdata('form') for _n in context])
            data = [doc.uid(node), node.sdata('lemma'), what_is, pred, prob, context_text]
            print('\t'.join([str(i) for i in data]))