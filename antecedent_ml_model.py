import warnings
from typing import Dict, List

import sklearn
import pandas as pd
import pickle
import random

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


data_df = pd.read_csv('./antecedent_dataset_balanced_dist.csv', sep='\t', encoding='utf-8')

columns = ['antec', 'licenser', 'is_cataphoric', 'cl_distance', 'node_id',
       'nsp_score', 'same_mod', 'same_mod_class', 'l_mod_class',
       'subj_imperat', 'is_rel', 'y']

variables = ['is_cataphoric', 'cl_distance', 'node_id',
       'nsp_score', 'same_mod', 'same_mod_class', 'l_mod_class',
       'subj_imperat', 'is_rel']
y_name = 'y'

X = data_df.loc[:,variables]
row_count = X.shape[0]
test_rows = random.sample(range(0, row_count), int(row_count/4))
train_rows = [i for i in range(0, row_count) if i not in test_rows]
X_train = X.iloc[train_rows]
X_test = X.iloc[test_rows]
y = data_df[y_name]
y_train = y[train_rows]
y_test = y[test_rows]

classifier_classes = {  KNeighborsClassifier: {'n_neighbors':3 },
                        MLPClassifier: {},
                        LogisticRegression: {'max_iter':500, 'class_weight':'balanced'},
                        LinearDiscriminantAnalysis : {},
                        QuadraticDiscriminantAnalysis : {}
}

classifier_scores = {}

for Classifier_class, args in classifier_classes.items():
    c_name = Classifier_class.__name__
    classifier_scores[c_name] = {}
    scores = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier = Classifier_class(**args).fit(X_train, y_train)
        score_train = classifier.score(X_train, y_train)
        score_test = classifier.score(X_test, y_test)
        classifier = Classifier_class(**args).fit(X, y)
        score_all = classifier.score(X, y)
    classifier_scores[c_name] = (score_train, score_test, score_all)
    print('\t'.join([c_name] + [str(f) for f in classifier_scores[c_name]]))
    classifier_scores[c_name] = (score_all, classifier)
