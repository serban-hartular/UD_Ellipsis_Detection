from typing import List, Dict

import stanza
import tree_path as tp
from stanza.utils.conll import CoNLL

nlp = stanza.Pipeline('ro', tokenize_pretokenized=True)

def wordlist_to_ParsedDoc(word_list : List[str], parser) -> tp.ParsedDoc:
    stanza_doc = nlp([word_list])
    CoNLL.write_doc2conll(stanza_doc, "temp.conllu")
    dl = tp.DocList.from_conllu("./temp.conllu")
    doc = dl[0]
    return doc

def wordlist_to_nodelist(word_list : List[str], parser) -> List[tp.Tree]:
    stanza_doc = nlp([word_list])
    CoNLL.write_doc2conll(stanza_doc, "temp.conllu")
    dl = tp.DocList.from_conllu("./temp.conllu")
    nodelist = []
    for doc in dl:
        nodelist.extend(list(doc.token_iter()))
    return nodelist

