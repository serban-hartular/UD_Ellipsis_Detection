
import tree_path as tp
from robert_client import RoBERT_Client

client = RoBERT_Client()

sentence = 'A încercat să nu antreneze golani, tineri care să arate pe stradă ce au învățat în sală. Nu întotdeauna a și reușit.'
word_vecs = client.get_vectors(sentence)
words = word_vecs['words']
for i in range(1, len(words)-1):
    text1 = ' '.join(words[:i])
    text2 = ' '.join(words[i:])
    print(' | '.join([text1, text2]), end='\t')
    print(client.get_next_sentence_probability(text1, text2))
