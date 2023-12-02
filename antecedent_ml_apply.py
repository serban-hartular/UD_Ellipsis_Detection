from typing import Dict, List

import pandas as pd

import tree_path as tp
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import pickle


import modality_ml_test
from antecedent_clause_test import split_sentence_to_clauses, clause_to_sentence_text, is_verb
from antecedent_generate_data import node_wvector_dict, modality_detector, mod_class_dict, is_subj_imperative, is_rel
from robert_client import RoBERT_Client

def get_antecedent_candidates(licenser : tp.Tree, doc : tp.ParsedDoc, client : RoBERT_Client,
                              wvector_dict : Dict[str, List[float]] = None,
                              antecedent : tp.Tree|None = None,
                              BEFORE = 5, AFTER = 3) -> List[Dict[str, str|float]]:
    if wvector_dict is None: wvector_dict = node_wvector_dict(doc, client)
    clauses = []  # split_sentence_to_clauses(_s, doc) for _s in doc]
    for sentence in doc:
        clauses.extend(split_sentence_to_clauses(sentence, doc))
    licenser_index = clauses.index([cl for cl in clauses if licenser in cl][0])
    text_ellipsis_cl = clause_to_sentence_text(clauses[licenser_index])
    licenser_modality = modality_ml_test.get_labels_from_word_vec(wvector_dict[doc.uid(licenser)], modality_detector)
    candidate_data = []
    candidate_clauses_enum = list(enumerate(clauses))[min(licenser_index - BEFORE, 0):licenser_index + AFTER]
    is_cataphoric = False
    correct_antecedent_row = None
    for candidate_clause_index, candidate_clause in candidate_clauses_enum:
        text_candidate_cl = clause_to_sentence_text(candidate_clause)
        for i, candidate in enumerate(candidate_clause):
            if candidate == licenser:
                is_cataphoric = True
                continue
            if not is_verb.find(candidate): continue  # not verb
            delta = candidate_clause_index - licenser_index
            row = {'antec': doc.uid(candidate),
                   'licenser': doc.uid(licenser),
                   'is_cataphoric': is_cataphoric,
                   'cl_distance': abs(delta),
                   'node_id': int(candidate.sdata('id')) / 100 * (-1 if is_cataphoric else 1)
                   }
            # next sentence probability
            text1, text2 = (text_ellipsis_cl, text_candidate_cl) if is_cataphoric else (
            text_candidate_cl, text_ellipsis_cl)
            nsp_score = client.get_next_sentence_probability(text1, text2)
            row['nsp_score'] = nsp_score
            # modalities
            candidate_parent = candidate.parent
            if candidate_parent is None:
                candidate_modality = ('', '')
            else:
                candidate_modality = modality_ml_test.get_labels_from_word_vec(wvector_dict[doc.uid(candidate)],
                                                                               modality_detector)
            row['same_mod'] = licenser_modality[0] == candidate_modality[0]
            row['same_mod_class'] = licenser_modality[1] == candidate_modality[1]
            row['l_mod_class'] = mod_class_dict[licenser_modality[1]]
            row['subj_imperat'] = bool(is_subj_imperative.find(candidate))
            row['is_rel'] = bool(is_rel.find(candidate))
            # result
            if antecedent:
                row['y'] = bool(candidate == antecedent)
            # to float
            row = {k: (v if isinstance(v, str) else float(v)) for k, v in row.items()}
            # store correct antecedent row, for balance
            if antecedent and row['y']:
                correct_antecedent_row = row
            # add to candidate
            candidate_data.append(row)
    # now I've got my candidate data, I will balance it if I know the antecedent
    if correct_antecedent_row:
        candidate_data.extend([correct_antecedent_row] * (len(candidate_data) - 1))
    return candidate_data


with open('./antecedent_classifiers-1-dbalanced.p', 'rb') as handle:
    classifier_dict = pickle.load(handle)

classifier = classifier_dict['KNeighborsClassifier'][1]
variables = ['is_cataphoric', 'cl_distance', 'node_id',
       'nsp_score', 'same_mod', 'same_mod_class', 'l_mod_class',
       'subj_imperat', 'is_rel']


dl = tp.DocList.from_conllu('./cancan21-annot-2.4-vpe.conllu')
client = RoBERT_Client()

for doc in dl:
    vpe = [m.node for m in doc.search('.//[misc.Target=unknown]')]
    if not vpe:
        continue
    for licenser in vpe:
        candidates = get_antecedent_candidates(licenser, doc, client)
        candidate_uids = [r['antec'] for r in candidates]
        df = pd.DataFrame(candidates)
        X = df[variables]
        y = classifier.predict_proba(X)
        results = list(zip(candidate_uids, [p[1] for p in y]))
        results.sort(key=lambda t: -t[1])
        print(results[0])


