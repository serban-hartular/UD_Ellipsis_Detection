import sklearn
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_pickle('./modality_data_rrt-1.p')
y_columns = ['modality_label', 'paramodality_label']
X = df.loc[:,'D0':'D767']

classifier_classes = {  KNeighborsClassifier: {'n_neighbors':1},
                        # MLPClassifier: {},
                        # GaussianNB : {}
}

classifier_scores = {}

for Classifier_class, args in classifier_classes.items():
    c_name = Classifier_class.__name__
    classifier_scores[c_name] = {}
    for y_name in y_columns:
        classifier_scores[c_name][y_name] = 0.0
        y = df[y_name]
        classifier = Classifier_class(**args).fit(X, y)
        score = classifier.score(X, y)
        classifier_scores[c_name][y_name] = (score, classifier)
        print('\t'.join([c_name, y_name, str(score)]))

with open('./modality_classifiers-1.p', 'wb') as handle:
    pickle.dump(classifier_scores, handle)
