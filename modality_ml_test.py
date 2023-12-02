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

def get_modalities_from_word_vec(vec : List[float], modality_detectors : Dict[str, Tuple],
                                 return_best = True)\
        -> List[str]:
    modalities = []
    probabilities = []
    for modality, (score, classifier) in modality_detectors.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = classifier.predict([vec])[0]
        probabilities.append(prob)
        if prob > 0.5:
            modalities.append(modality)
    if not modalities and return_best:
        prob_mod = [t for t in zip(modality_detectors.keys(), probabilities)]
        prob_mod.sort(key=lambda t: t[1])
        modalities = [prob_mod[-1][0]+'?']
    return modalities

def get_labels_from_word_vec(vec : List[float], modality_detectors : Dict[str, Tuple]) -> Tuple[str, str]:
    labels = []
    for modality, (score, classifier) in modality_detectors.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = classifier.predict([vec])[0]
        labels.append(pred)
    return tuple(labels)

# client = RoBERT_Client('http://127.0.0.1:5001')
VECTOR_LEN = 768

# dl = tp.DocList.from_conllu('./rrt-all.3.annot.5.2.conllu')
# doc = dl[0]

with open('./modality_classifiers-1.p', 'rb') as handle:
    modality_detectors = pickle.load(handle)

# for sentence in doc:
#     vpe = [m.node for m in tp.Search('.//[misc.Ellipsis=VPE]').find(sentence)]
#     if not vpe: continue
#     tokens = sentence.projection_nodes()
#     encodings = client.get_vectors([t.sdata('form') for t in tokens])
#     word_vectors = encodings['word_vectors']
#     for node in vpe:
#         fixed = [m.node for m in tp.Search('/[deprel=fixed]').find(node)]
#         if fixed: node = fixed[-1]
#         vector = word_vectors[int(node.sdata('id'))-1]
#         labels = get_labels_from_word_vec(vector, modality_detectors['KNeighborsClassifier'])
#         data = [doc.uid(node), node.sdata('lemma')] + labels + [sentence.sent_text]
#         print('\t'.join(data))

