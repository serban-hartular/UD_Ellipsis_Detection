import collections
import itertools
from typing import List, Tuple

import tree_path as tp
from robert_client import RoBERT_Client

is_part_of_predicate = tp.Search(
    '.[(deprel=ccomp,ccomp:pmod,csubj,csub:pass) | (deprel=xcomp & (upos=VERB | /[deprel=cop]) )]'
)
is_verb = tp.Search('.[upos=VERB | /[deprel=cop] ]')

Clause = List[tp.Tree]

def clause_root(cl : Clause) -> tp.Tree:
    node = cl[0]
    while node.parent in cl:
        node = node.parent
    return node

def clause_splitter(root : tp.Tree, doc : tp.ParsedDoc) -> (Clause, List[Clause]):
    this_clause = [root]
    other_clauses = []
    for child in root.children():
        child_clause, child_others = clause_splitter(child, doc)
        if is_verb.find(child) and not is_part_of_predicate.find(child):
            other_clauses.append(child_clause)
        else:
            this_clause.extend(child_clause)
        other_clauses += child_others
    return this_clause, other_clauses


def split_sentence_to_clauses(sentence : tp.Tree|str, doc : tp.ParsedDoc) -> List[Clause]:
    """Returns lists of uids corresponding to the contents of each clause"""
    if isinstance(sentence, str):
        sentence = doc.get_node_by_uid(sentence)
    root_clause, other_clauses = clause_splitter(sentence, doc)
    clauses = [root_clause] + other_clauses
    # sort clauses by first uid, which is the root's
    clauses.sort(key=lambda _node_list: int(_node_list[0].sdata('id')))
    # then sort each individual clause by id
    for clause in clauses:
        clause.sort(key=lambda _node: int(_node.sdata('id')))
    return clauses

def node_projection_text(node : tp.Tree) -> str:
    return ' '.join([t.sdata('form')
                     for t in node.projection_nodes()])

def node_list_text(node_list : List[tp.Tree]) -> str:
    return ' '.join([_node.sdata('form') for _node in node_list])

def clause_to_sentence_text(clause : Clause) -> str:
    nodes = list(clause) # copy, cause we're going to mess with it.
    _to_eliminate_before = tp.Search(
        '.[(deprel=mark & !(lemma=să | feats.PartType=Inf)) | deprel=cc,punct | (deprel=case & ../[upos=VERB] ) | feats.PronType=Rel | /[deprel=fixed feats.PronType=Rel] | (deprel=fixed & ../[/[deprel=fixed feats.PronType=Rel]] ) ]')
    _to_eliminate_after = tp.Search('.[deprel=punct]')
    for i, node in enumerate(nodes):
        if not _to_eliminate_before.find(node):
            break
    nodes = nodes[i:]
    for i, node in reversed(list(enumerate(nodes))):
        if not _to_eliminate_after.find(node):
            break
    nodes = nodes[:i+1]
    words = [n.sdata('form') for n in nodes]
    words[0] = words[0][0].upper() + words[0][1:]
    words = ' '.join(words + ['.'])
    return words

# CandidateClauses = collections.namedtuple('CandidateClauses', ['elliptic', 'before', 'after'])

def generate_candidate_clauses(licenser : tp.Tree, doc : tp.ParsedDoc, before : int = 6, after : int = 3)\
        -> (Clause, List[Clause], List[Clause]):
    ellipsis_sentence : tp.ParsedSentence = licenser.root()
    ellipsis_sentence_index = doc.index(ellipsis_sentence)
    current_clauses = split_sentence_to_clauses(ellipsis_sentence, doc)
    elliptic_clause = [cl for cl in current_clauses if licenser in cl]
    elliptic_clause = elliptic_clause[0]
    elliptic_clause_index = current_clauses.index(elliptic_clause)
    before_clauses = list(reversed(current_clauses[:elliptic_clause_index]))
    after_clauses = current_clauses[elliptic_clause_index+1:]
    for sentence in doc[ellipsis_sentence_index+1:]:
        if len(after_clauses) >= after:
            break
        after_clauses.extend(split_sentence_to_clauses(sentence, doc))
    for sentence in reversed(doc[:ellipsis_sentence_index]):
        if len(before_clauses) >= before:
            break
        before_clauses.extend(reversed(split_sentence_to_clauses(sentence, doc)))
    return elliptic_clause, before_clauses[:before], after_clauses[:after]

def guess_next_clause(elliptic_clause, before_clauses, after_clauses, client : RoBERT_Client) \
        -> (List[Tuple[Clause, float]], List[Tuple[Clause, float]]):
    clause_texts = [clause_to_sentence_text(cl) for cl in [elliptic_clause] + before_clauses + after_clauses]
    elliptic_text, before_texts, after_texts = clause_texts[0], clause_texts[1:len(before_clauses)+1], clause_texts[len(before_clauses)+1:]
    before_scores = [(cl, client.get_next_sentence_probability(cl_text, elliptic_text)) for cl, cl_text in zip(before_clauses, before_texts)]
    after_scores = [(cl, client.get_next_sentence_probability(elliptic_text, cl_text)) for cl, cl_text in zip(after_clauses, after_texts)]
    return before_scores, after_scores

if __name__ == "__main__":
    client = RoBERT_Client('http://127.0.0.1:5001')
    dl = tp.DocList.from_conllu('./cancan21-annot-2.2-vpe.conllu')
    for doc in dl:
        vpe = doc.search('.//[misc.Ellipsis=VPE misc.Antecedent=Present]')
        for licenser in [m.candidate for m in vpe]:
            if licenser.sdata('misc.TargetID'):
                antecedent = doc.get_node_by_uid(licenser.sdata('misc.TargetID'))
            else:
                continue
            elliptic_clause, before_clauses, after_clauses = generate_candidate_clauses(licenser, doc)
            before_scores, after_scores = guess_next_clause(elliptic_clause, before_clauses, after_clauses, client)
            all_scores = before_scores # + after_scores
            all_scores.sort(key=lambda t : -t[1])
            position = -1
            for i, (cl, score) in enumerate(all_scores):
                if antecedent in cl:
                    position = i
                    break
            print(doc.uid(licenser), position)

