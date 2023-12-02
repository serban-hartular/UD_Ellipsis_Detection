import warnings

import sklearn
import pandas as pd
import pickle
import random

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_pickle('./ellipticity_data_df.p')
class_labels = ['use_type']
binary_labels = ['incomplete', 'elliptic']
X = df.loc[:,'D0':'D767']
row_count = X.shape[0]
test_rows = random.sample(range(0, row_count), int(row_count/4))
train_rows = [i for i in range(0, row_count) if i not in test_rows]
X_train = X.iloc[train_rows]
X_test = X.iloc[test_rows]

classifier_classes = {  KNeighborsClassifier: {'n_neighbors':3 },
                        MLPClassifier: {},
#                        GaussianNB : {}
}

discriminator_classes = {
        LogisticRegression: {'max_iter':500, 'class_weight':'balanced'},
        LinearDiscriminantAnalysis : {},
        QuadraticDiscriminantAnalysis : {}
}

classifier_scores = {}

for Classifier_class, args in classifier_classes.items():
    c_name = Classifier_class.__name__
    classifier_scores[c_name] = {}
    for y_name in class_labels + binary_labels:
        classifier_scores[c_name][y_name] = 0.0
        y = df[y_name]
        y_train = y[train_rows]
        y_test = y[test_rows]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier = Classifier_class(**args).fit(X_train, y_train)
            classifier_all = Classifier_class(**args).fit(X_train, y_train)
        score = classifier.score(X_train, y_train)
        test_score = classifier.score(X_test, y_test)
        classifier_scores[c_name][y_name] = (test_score, classifier_all)
        print('\t'.join([c_name, y_name, str(score), str(test_score)]))

for Classifier_class, args in discriminator_classes.items():
    c_name = Classifier_class.__name__
    classifier_scores[c_name] = {}
    for y_name in binary_labels:
        classifier_scores[c_name][y_name] = 0.0
        y = df[y_name]
        y_train = y[train_rows]
        y_test = y[test_rows]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier = Classifier_class(**args).fit(X_train, y_train)
            classifier_all = Classifier_class(**args).fit(X_train, y_train)
        score = classifier.score(X_train, y_train)
        test_score = classifier.score(X_test, y_test)
        classifier_scores[c_name][y_name] = (test_score, classifier_all)
        print('\t'.join([c_name, y_name, str(score), str(test_score)]))

with open('./ellipticity_detection.p', 'wb') as handle:
    pickle.dump(classifier_scores, handle)
