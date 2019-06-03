
DATASETS = ['tweets', 'manifesto', 'reddit', 'smoking', 'immigration', 'samesex', 'manifestocoarse', 'mh17', 'smokingimmigrationsamesex',
            'smokingimmigrationsamesextweets', 'dialogue', 'dialogueslidermean', 'dialoguesliderdev', 'dialoguepairtype']

TASK2EMBS = {'tweets': 'glove',
             'smoking': 'glove',
             'immigration_1': 'glove',
             'samesex_1': 'glove',
             'immigration_2': 'glove',
             'samesex_2': 'glove',
             'immigration': 'glove',
             'samesex': 'glove',
             'manifesto': 'glove',
             'manifestocoarse':'glove',
             'reddit': 'glove',
             'smokingimmigrationsamesextweets': 'glove',
             'smokingimmigrationsamesex':'glove',
             'dialogue':'glove',
             'default':'glove',
             'dialogueslidermean':'glove',
            'dialoguesliderdev' :'glove',
            'dialoguepairtype':'glove'}

def get_emb_name_for_task(task):
    if task in TASK2EMBS.keys():
        return TASK2EMBS[task]
    else:
        return TASK2EMBS['default']

