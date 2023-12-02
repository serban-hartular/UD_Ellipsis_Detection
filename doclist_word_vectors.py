from typing import Dict, List

import tree_path as tp
from robert_client import RoBERT_Client
import pickle

def doclist2wvectors(src : tp.DocList|str, client : RoBERT_Client) -> Dict[str, List[float]]:
    """Take doclist or path to conllu and return dict of word vectors for each uid"""
    if isinstance(src, str):
        src = tp.DocList.from_conllu(src)
    vector_dict = {}
    for i, doc in enumerate(src):
        print('Doc %d of %d' % (i+1, len(src)))
        sentences = [_s.projection_nodes() for _s in doc]
        for i, sentence in enumerate(sentences):
            if i == 0:
                nodes = sentences[0] + (sentences[1] if len(sentences) > 1 else [])
                start_index = 0
            else:
                nodes = sentences[i - 1] + sentences[i] + (sentences[i + 1] if len(sentences) > (i + 1) else [])
                start_index = len(sentences[i - 1])
            words = [_n.sdata('form') for _n in nodes]
            vectors = client.get_vectors(words)
            node_vectors = zip(nodes[start_index:start_index + len(sentence)],
                               vectors['word_vectors'][start_index:start_index + len(sentence)])
            node_vectors = {doc.uid(_n): vec for _n, vec in node_vectors}
            if set(node_vectors.keys()).intersection(set(vector_dict.keys())):
                raise Exception("Duplicate uids: %s" % str(set(node_vectors.keys()).intersection(set(vector_dict.keys()))))
            vector_dict.update(node_vectors)
    return vector_dict

client = RoBERT_Client()
vector_dict = doclist2wvectors('./wowbiz21-annot-1.2-vpe.conllu', client)
with open('./wowbiz21-annot-1.2-vpe.vecs.p', 'wb') as handle:
    pickle.dump(vector_dict, handle)
