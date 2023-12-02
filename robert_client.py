from typing import List, Tuple, Dict

import requests

def request_post_catch_timeout(url : str, data : Dict|List, timeout : float):
    try:
        r = requests.post(url, json=data, timeout=timeout)
    except Exception as e:
        raise Exception('Communication error, url = "%s"' % url)
    try:
        r = r.json()
    except Exception as e:
        raise Exception('Error converting to json: "%s"' % str(e))
    return r


class RoBERT_Client:
    def __init__(self, url : str = 'http://127.0.0.1:5001', timeout : float = 10):
        self.url = url
        self.timeout = timeout
        # r = requests.post(self.url, json={'ping':'ping'}, timeout=self.timeout)
        # r = r.json()
        r = request_post_catch_timeout(url, {'ping':'ping'}, self.timeout)
        if r.get('status') != 'ok':
            raise Exception('Could not ping url %s: %s' % (url, str(r)))
        print(r)
    def get_vectors(self, text : str | List[str]) -> dict:
        # r = requests.post(self.url, json={'text':text})
        # return r.json()
        r = request_post_catch_timeout(self.url, {'text':text}, self.timeout)
        return r

    def get_next_sentence_probability(self, text1 : str, text2 : str) -> float:
        # r = requests.post(self.url + '/nsp', json={'text1':text1, 'text2':text2}, timeout=self.timeout)
        # r = r.json()
        r = request_post_catch_timeout(self.url + '/nsp', {'text1':text1, 'text2':text2}, self.timeout)
        if 'probability' in r:
            return float(r['probability'])
        raise Exception('Error: %s' % (str(r)))

    def predict_masked_words(self, text : str, num_results : int = 5) \
            -> List[List[Tuple[str, float]]]:
        """Returns, for each masked word, a list of (word_form, weight)
        list length is num_results"""
        # r = requests.post(self.url + '/mask',
        #                   json={'text': text, 'num_results': num_results}, timeout=self.timeout)
        # r = r.json()
        r = request_post_catch_timeout(self.url + '/mask',
                                       {'text': text, 'num_results': num_results}, self.timeout)
        if r['status'] != 'ok':
            raise Exception('error: ' + str(r))
        results = r['results']
        try:
            tuple_results = [[tuple(option) for option in mask] for mask in results]
        except Exception as e:
            raise Exception('Error in result data format of "%s": %s' % (str(results), str(e)))
        return tuple_results

