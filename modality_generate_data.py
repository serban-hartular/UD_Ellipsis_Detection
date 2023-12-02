import tree_path as tp
import pandas as pd
import sklearn

from robert_client import RoBERT_Client

lemma_mods = pd.read_csv('./lemma_modalities.csv', sep='\t', encoding='utf-8')
lemma_mods = lemma_mods.to_dict(orient="records")
lemma_mods = {(rec['lemma'], rec['upos']):rec['modality'].split('|') for rec in lemma_mods}
modalities = set()
for v in lemma_mods.values():
    modalities.update(set(v))
modalities = list(modalities)
modalities.sort()
modality_index = {m:i for i,m in enumerate(modalities)}

client = RoBERT_Client('http://127.0.0.1:5001')
VECTOR_LEN = 768

dl = tp.DocList.from_conllu('./rrt-all.3.annot.5.2.conllu')
doc = dl[0]

para_modalities_dict = {
    'DV'    : 'ParaDeont',
    'N'     : 'None',
    'EVID'  : 'ParaEpist',
    'ASP'   : 'ParaDeont',
    'EPIST' : 'ParaEpist',
    'CAUS'  : 'ParaDeont',
    'DV|ASP': 'ParaDeont',
    'APREC' : 'ParaDeont'
}

data = []
for sentence in doc:
    tokens = sentence.projection_nodes()
    word_vectors = None
    for i, node in enumerate(tokens):
        uid = doc.uid(node)
        mods = lemma_mods.get((node.sdata('lemma'), node.sdata('upos')))
        if mods is None:
            continue
        if list(tp.Search('/[lemma=sine feats.Strength=Weak]').find(node)):
            continue
        if word_vectors is None:
            word_vectors = client.get_data([t.sdata('form') for t in tokens])['word_vectors']
        modality_flags = [0.0]*len(modalities)
        for m in mods:
            modality_flags[modality_index[m]] = 1.0
        row = [uid] + word_vectors[i] + modality_flags
        # modality and paramodality labels
        mods = '|'.join(mods)
        row.append(mods)
        row.append(para_modalities_dict[mods])
        # append row
        print(uid, row[-10:])
        data.append(row)


df = pd.DataFrame(data, columns=['uid'] + ['D%d'%i for i in range(VECTOR_LEN)]+list(modalities)
                  + ['modality_label', 'paramodality_label'])
X = df.loc[:,'D0':'D767']