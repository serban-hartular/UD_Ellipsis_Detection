import robert_client
import tree_path as tp
import pandas as pd
import sklearn

from robert_client import RoBERT_Client

def full_lemma(node : tp.Tree, return_form = False) -> str:
    return ' '.join([node.sdata('form').lower() if return_form else node.sdata('lemma')] +
                    [m.node.sdata('lemma') for m in tp.Search('/[deprel=fixed]').find(node)])

dl = tp.DocList.from_conllu('./rrt-all.3.annot.5.2.conllu')
doc = dl[0]

vpe = [m.candidate for m in doc.search('.//[misc.Ellipsis=VPE]')]
vpe_full_lemmas = {(full_lemma(n, True), n.sdata('upos')) for n in vpe}

client = RoBERT_Client('http://127.0.0.1:5001')

data = []

for sentence in doc:
    vectors = None
    node_list = sentence.projection_nodes()
    for i, node in enumerate(node_list):
        lemma = full_lemma(node, True)
        if (lemma, node.sdata('upos')) not in vpe_full_lemmas:
            continue
        use_type = node.sdata('misc.Ellipsis')
        if not vectors:
            vectors = client.get_data([n.sdata('form') for n in node_list])
        if not use_type: use_type = 'Regular'
        incomplete_flag = int(use_type in ('VPE', 'RNR', 'Absolute', 'Semantic'))
        elliptic_flag = int(use_type in ('VPE', 'RNR'))
        if use_type in ('BadParse', 'WrongValence', 'Meaning'): use_type = 'Regular'
        row = [doc.uid(node), lemma.replace(' ', '_'), use_type, incomplete_flag, elliptic_flag] + vectors['word_vectors'][i]
        print(row[:5])
        data.append(row)
