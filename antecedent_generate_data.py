import itertools
from collections import defaultdict
from typing import Dict, List

import modality_ml_test
import tree_path as tp
from robert_client import RoBERT_Client
import pandas as pd
from antecedent_clause_test import split_sentence_to_clauses, Clause, clause_root, is_verb, clause_to_sentence_text

modality_detector = modality_ml_test.modality_detectors['KNeighborsClassifier']

def node_wvector_dict(doc : tp.ParsedDoc, client : RoBERT_Client) -> Dict[str, List[float]]:
    """return dict of uids and word vectors"""
    sentences = [_s.projection_nodes() for _s in doc]
    vector_dict = {}
    for i, sentence in enumerate(sentences):
        if i == 0:
            nodes = sentences[0] + (sentences[1] if len(sentences)>1 else [])
            start_index = 0
        else:
            nodes = sentences[i-1] + sentences[i] + (sentences[i+1] if len(sentences)>(i+1) else [])
            start_index = len(sentences[i-1])
        words = [_n.sdata('form') for _n in nodes]
        vectors = client.get_vectors(words)
        node_vectors = zip(nodes[start_index:start_index+len(sentence)], vectors['word_vectors'][start_index:start_index+len(sentence)])
        vector_dict.update({doc.uid(_n) : vec for _n, vec in node_vectors})
    return vector_dict


is_subj_imperative = tp.Search('.[feats.Mood=Imp | /[upos=PART lemma=să] ]')
is_rel = tp.Search('<[(feats.PronType=Rel !lemma=care) | /[feats.PronType=Rel !lemma=care] ]')

mod_class_dict = defaultdict(int, {'ParaEpist':1, 'ParaDeont' : 2})

class AntecedentDataGenerator:
    def generate_instance_datapoints(self) -> List[List[float|str]]:
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
                candidate_modality = modality_ml_test.get_labels_from_word_vec(vector_dict[doc.uid(candidate)],
                                                                               modality_detector)
            row['same_mod'] = licenser_modality[0] == candidate_modality[0]
            row['same_mod_class'] = licenser_modality[1] == candidate_modality[1]
            row['l_mod_class'] = mod_class_dict[licenser_modality[1]]
            row['subj_imperat'] = bool(is_subj_imperative.find(candidate))
            row['is_rel'] = bool(is_rel.find(candidate))
            # result
            row['y'] = bool(candidate == antecedent)
            # to float
            row = {k: (v if isinstance(v, str) else float(v)) for k, v in row.items()}


if __name__ == '__main__':
    data = []
    labels = set()

    dl = tp.DocList.from_conllu('./cancan21-annot-2.2-vpe.conllu')
    client = RoBERT_Client()
    BEFORE = 5
    AFTER = 3

    for doc in dl:
        vpe = [m.node for m in doc.search('.//[misc.Ellipsis=VPE misc.Antecedent=Present]')]
        if not vpe:
            continue
        vector_dict = node_wvector_dict(doc, client)
        clauses = [] #split_sentence_to_clauses(_s, doc) for _s in doc]
        for sentence in doc:
            clauses.extend(split_sentence_to_clauses(sentence, doc))
        for licenser in vpe:
            licenser_index = clauses.index([cl for cl in clauses if licenser in cl][0])
            text_ellipsis_cl = clause_to_sentence_text(clauses[licenser_index])
            licenser_modality = modality_ml_test.get_labels_from_word_vec(vector_dict[doc.uid(licenser)], modality_detector)
            antecedent = doc.get_node_by_uid(licenser.sdata('misc.TargetID'))
            antecedent_index = clauses.index([cl for cl in clauses if antecedent in cl][0])
            delta = antecedent_index - licenser_index # negative if anaphoric, positive if cataphoric
            if delta < -BEFORE or delta > AFTER:
                print('Antecedent of %s beyond before and after' % doc.uid(licenser))
                continue
            candidate_data = []
            candidate_clauses_enum = list(enumerate(clauses))[min(licenser_index-BEFORE,0):licenser_index+AFTER]
            is_cataphoric = False
            correct_antecedent_row = None
            for candidate_clause_index, candidate_clause in candidate_clauses_enum:
                text_candidate_cl = clause_to_sentence_text(candidate_clause)
                for i, candidate in enumerate(candidate_clause):
                    if candidate == licenser:
                        is_cataphoric = True
                        continue
                    if not is_verb.find(candidate): continue # not verb
                    delta = candidate_clause_index - licenser_index
                    row = { 'antec' : doc.uid(candidate),
                            'licenser' : doc.uid(licenser),
                            'is_cataphoric' : is_cataphoric,
                            'cl_distance': abs(delta),
                            'node_id' : int(candidate.sdata('id')) / 100 * (-1 if is_cataphoric else 1)
                    }
                    # next sentence probability
                    text1, text2 = (text_ellipsis_cl, text_candidate_cl) if is_cataphoric else (text_candidate_cl, text_ellipsis_cl)
                    nsp_score = client.get_next_sentence_probability(text1, text2)
                    row['nsp_score'] = nsp_score
                    # modalities
                    candidate_parent = candidate.parent
                    if candidate_parent is None:
                        candidate_modality = ('', '')
                    else:
                        candidate_modality = modality_ml_test.get_labels_from_word_vec(vector_dict[doc.uid(candidate)], modality_detector)
                    row['same_mod'] = licenser_modality[0] == candidate_modality[0]
                    row['same_mod_class'] = licenser_modality[1] == candidate_modality[1]
                    row['l_mod_class'] = mod_class_dict[licenser_modality[1]]
                    row['subj_imperat'] = bool(is_subj_imperative.find(candidate))
                    row['is_rel'] = bool(is_rel.find(candidate))
                    # result
                    row['y'] = bool(candidate == antecedent)
                    # to float
                    row = {k:(v if isinstance(v, str) else float(v)) for k,v in row.items()}
                    # store correct antecedent row, for balance
                    if row['y']:
                        correct_antecedent_row = row
                    # add to candidate
                    candidate_data.append(row)
                    print('\t'.join([str(i) for i in row.values()]))
                    labels.update(row.keys())
            # now I've got my candidate data, I will balance it
            if correct_antecedent_row:
                candidate_data.extend([correct_antecedent_row]*(len(candidate_data)-1))
            data.extend(candidate_data)


    df = pd.DataFrame(data)
    # df.to_csv('./antecedent_dataset.csv', sep='\t', encoding='utf-8')
