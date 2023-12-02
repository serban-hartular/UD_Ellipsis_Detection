import pickle
import random
import warnings
from collections import defaultdict, namedtuple
from typing import List, Dict, Tuple

import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin

class BaseModel(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        pass

Scores = namedtuple('Scores', 'train, test, all')

class DatasetTrainer:
    def __init__(self, data_df : pd.DataFrame, variables : List[str], outcomes : List[str],
                 model_classes : List[Tuple[BaseModel.__class__, Dict]],
                 test_train_split : float = 0.25, **kwargs):
        debugFlag = bool(kwargs.get('debug'))
        self._meta_data = kwargs.get('meta_data')
        self._variables = variables
        self._outcomes = outcomes
        self._model_classes = model_classes
        X = data_df.loc[:,variables]
        row_count = X.shape[0]
        test_rows = random.sample(range(0, row_count), int(row_count*test_train_split))
        train_rows = [i for i in range(0, row_count) if i not in test_rows]
        X_train, X_test = X.iloc[train_rows], X.iloc[test_rows]
        model_dict = defaultdict(defaultdict)
        for y_name in self._outcomes:
            y = data_df[y_name]
            y_train, y_test = y[train_rows], y[test_rows]
            for ModelClass, args in model_classes:
                if debugFlag: print('%s %s' % (ModelClass.__name__, str(args)), end='\t')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    classifier = ModelClass(**args).fit(X_train, y_train)
                    score_train = classifier.score(X_train, y_train)
                    score_test = classifier.score(X_test, y_test)
                    classifier = ModelClass(**args).fit(X, y)
                    score_all = classifier.score(X, y)
                scores = Scores(score_train, score_test, score_all)
                model_dict[y_name][(ModelClass, tuple(args.items()))] = (classifier, scores)
                if debugFlag: print(scores)

        self._model_dict = dict(model_dict)

    def get_classifier(self, model : type|str|Tuple[type, Tuple] = None, outcome : str = None)\
            -> BaseModel:
        return self.get_classifier_wscores(model, outcome)[0]

    def get_classifier_wscores(self, model : type|str|Tuple[type, Dict] = None, outcome : str = None)\
            -> (BaseModel, Scores):
        if not outcome:
            outcome = self._outcomes[0]
        if not model:
            model = self._model_classes[0]
        if isinstance(model, Tuple):
            _class, _args = model
            _args = tuple(_args.items())
            return self._model_dict[outcome][(_class, _args)]
        if isinstance(model, type):
            model = model.__name__
        for (_class, _args), _classifier in self._model_dict[outcome].items():
            if _class.__name__ == model:
                return _classifier
        return (None, None)

    def to_pickle(self, filename : str):
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle)

    @staticmethod
    def from_pickle(filename) -> 'DatasetTrainer':
        obj = None
        with open(filename, 'rb') as handle:
            obj = pickle.load(handle)
        return obj

print('Loading data...')
data_df = pd.read_csv('./antecedent_dataset_balanced_dist.csv', sep='\t', encoding='utf-8')

columns = ['antec', 'licenser', 'is_cataphoric', 'cl_distance', 'node_id',
       'nsp_score', 'same_mod', 'same_mod_class', 'l_mod_class',
       'subj_imperat', 'is_rel', 'y']

variables = ['is_cataphoric', 'cl_distance', 'node_id',
       'nsp_score', 'same_mod', 'same_mod_class', 'l_mod_class',
       'subj_imperat', 'is_rel']
y_name = 'y'

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

classifier_classes = [  (KNeighborsClassifier, {'n_neighbors':3 }),
                        #(MLPClassifier, {}),
                        (LogisticRegression, {'max_iter':500, 'class_weight':'balanced'}),
                        (LinearDiscriminantAnalysis, {}),
                        (QuadraticDiscriminantAnalysis, {}),
]
# dtrainer = DatasetTrainer(data_df, variables, [y_name], classifier_classes,
#                           debug=True, meta_data='Antecedent detection')

