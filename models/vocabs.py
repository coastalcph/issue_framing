from models import utils
import numpy as np

class Char2Int():

    def __init__(self, startSym='', endSym='', unkSym='UNK'):
        self.startSym = startSym
        self.endSym = endSym
        self.unkSym = unkSym
        self.char2int = {}
        self.int2char = {}
        self.vocab_size = 0

    def build_vocab(self, seqs):
        characters = {}
        for seq in seqs:
            for c in seq:
                characters.setdefault(c)
        characters.setdefault(self.startSym)
        characters.setdefault(self.endSym)
        characters.setdefault(self.unkSym)
        characters = list(characters.keys())
        self.int2char = characters
        self.char2int = {c: i for i, c in enumerate(characters)}
        self.vocab_size = len(characters)

    def transform_char2int(self, seq):
        # append start and end symbol
        seq = [c for c in seq]
        if self.startSym != '':
            seq = [self.startSym] + seq
        if self.endSym != '':
            seq = seq + [self.endSym]
        seq = [c if c in self.char2int else self.unkSym for c in seq]
        return [self.char2int[c] for c in seq]


class Word2Int():
    def __init__(self, startSym='', endSym='', unkSym='UNK', embeds_file=None):
        self.word2int = {}
        self.int2word = {}
        self.vocab_size = 0
        self.unk = unkSym
        self.startSym = startSym
        self.endSym = endSym
        self.embeds_file = embeds_file
        self.embeds = None

    # tokenize the sequence by splitting at white space
    def tokenize(self, seq):
        return seq.split(' ')

    def build_vocab(self, seqs):
        types = {}
        for seq in seqs:
            for t in self.tokenize(seq):
                types.setdefault(t)
        types.setdefault(self.unk)
        if self.startSym != '':
            types.setdefault(self.endSym)
        if self.endSym != '':
            types.setdefault(self.startSym)
        words = list(types.keys())

        # build the vocabulary based on the pretrained embeddings
        # add all of the pretrained embedding words to the vocabulary
        if self.embeds_file:
            word2vec, dim = utils.load_wordembedds(self.embeds_file)
            word2_vec_keys = set(word2vec.keys())
            words = list(set(word2vec.keys()).union(set(words)))
            num_of_words = len(words)
            self.embeds = np.zeros((num_of_words, dim))
            for i in range(len(words)):
                word = words[i]
                if word in word2_vec_keys:
                    self.embeds[i, :] = word2vec[word]

        self.int2word = words
        self.word2int = {w: i for i, w in enumerate(words)}
        self.word2int_keys = set(self.word2int.keys())
        self.vocab_size = len(words)


    def save(self, fname):
        fname = fname + '.vocab'
        vocab = {'embeds_file': self.embeds_file, 'word2int': self.word2int, 'endSym': self.endSym,'startSym': self.startSym, 'vocab_size':self.vocab_size}
        utils.write_json(vocab, fname)

    def load(self, fname):
        fname = fname +'.vocab'
        vocab = utils.load_json(fname)
        self.word2int = vocab['word2int']
        self.endSym = vocab['endSym']
        self.startSym = vocab['startSym']
        self.word2int_keys = set(self.word2int.keys())
        self.vocab_size = vocab['vocab_size']
        

    def transform_word2int(self, seq):
        # replace unknown words by unk token
        seq = self.tokenize(seq)
        # append start and end symbol
        if self.startSym != '':
            seq = [self.startSym] + seq
        if self.endSym != '':
            seq = seq + [self.endSym]
        seq = [word if word in self.word2int_keys else self.unk for word in seq]
        return [self.word2int[w] for w in seq]

    def check_for_unseen_words(self, seq):
        for elm in self.tokenize(seq):
            if elm not in self.word2int:
                print('Word: {} is unknown'.format(elm))

    def get_int2word(self):
        '''
        return the index to word mapping. if it's empty, initialize it
        :return:
        '''
        if len(self.int2word.keys()) == 0:
            for word, int in self.word2int.items():
                self.int2word[int] = word
        return self.int2word




