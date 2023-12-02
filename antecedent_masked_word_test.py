import collections
import itertools
from typing import List, Tuple

import tree_path as tp
from robert_client import RoBERT_Client
import stanza_setup

is_verb = tp.Search('.[(upos=VERB & !deprel=aux,aux:pass,cop) | /[deprel=cop] ]')


def insert_masks(word_list : List[str], insert_after_index : int, num_masks : int = 1)\
        -> List[str]:
    """Returns list of words with mask tokens inserted after index"""
    word_list = word_list[:insert_after_index+1] + (['[MASK]']*num_masks) + word_list[insert_after_index+1:]
    return word_list

def reconstitute_masked(word_list : List[str], client : RoBERT_Client) -> List[str]:
    """change list in place, return index of first mask"""
    new_list = []
    masked_tokens = client.predict_masked_words(' '.join(word_list))
    # guesses = [mask[0][0] for mask in masked_list]
    for i, word in enumerate(word_list):
        if word != '[MASK]':
            new_list.append(word)
        else:
            guess = masked_tokens.pop(0)
            guess = guess[0][0]
            if guess.startswith('##'):
                guess = guess[2:]
                if i == 0:
                    new_list.append(guess)
                else:
                    new_list[-1] = new_list[-1] + guess
            else:
                new_list.append(guess)
    return new_list

def guess_ellided(sentence : tp.ParsedSentence, licenser : tp.Tree, client : RoBERT_Client)\
        -> (List[str], int):
    node_list = sentence.projection_nodes()
    insert_after_index = node_list.index(licenser)
    word_list = [n.sdata('form') for n in node_list]
    max_tries = 10
    new_node = None
    for _ in range(0, max_tries):
        word_list = insert_masks(word_list, insert_after_index) # insert 1 mask
        word_list = reconstitute_masked(word_list, client)
        insert_after_index += 1 # move to first guess
        new_parse = stanza_setup.wordlist_to_nodelist(word_list, stanza_setup.nlp)
        new_node = new_parse[insert_after_index]
        if is_verb.find(new_node):
            break
    return word_list, new_node


client = RoBERT_Client()

dl = tp.DocList.from_conllu('./rrt-all.3.annot.5.2.conllu')
for doc in dl:
    vpe = doc.search('.//[misc.Ellipsis=VPE misc.Antecedent=Present]')
    for licenser in [m.candidate for m in vpe]:
        if licenser.sdata('misc.TargetID'):
            antecedent = doc.get_node_by_uid(licenser.sdata('misc.TargetID'))
        else:
            continue
        sentence = licenser.root()
        wl, node = guess_ellided(sentence, licenser, client)
        print('\t'.join([antecedent.sdata('form'), node.sdata('form'),
                                                              ' '.join(wl)]))
