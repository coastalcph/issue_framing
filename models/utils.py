import numpy as np
import codecs
import random
import json
import os
import subprocess



def load_wordembedds(fname, sep=' ', lower=True):
    """
    Loads wordembeddings
    :param fname: name of embedding file
    :param sep: separator in embedding file
    :param lower: if True, lowercase all vocabulary words
    :return:
    """
    word2vec = dict()
    for line in open(fname):
        fields = line.split(sep)
        vec = [float(x) for x in fields[1:]]
        word = fields[0]
        if lower:
            word = word.lower()
        word2vec[word] = vec
    dim = len(vec)
    print('Loaded pre-trained embeddings for {} words. Dim= {}. (lower: {})'
          .format(len(list(word2vec.keys())), dim, lower))
    return word2vec, dim

def readFile(fName):
    with codecs.open(fName, 'r', 'utf-8') as f:
        content = f.readlines()
    content = [line.strip('\n') for line in content]
    return content

"""
def get_train_test_idxs_kfold(k, num_instances, train_frac, RANDOM_SEED):
    random.seed(RANDOM_SEED)
    idxs = range(num_instances)
    train = []
    test = []
    bidx = int(np.ceil(num_instances*train_frac))
    for i in range(k):
        random.shuffle(idxs)
        train.append(idxs[:bidx])
        test.append(idxs[bidx:])
    return train, test

def get_train_test_idxs(num_instances, train_frac, RANDOM_SEED):
    random.seed(RANDOM_SEED)
    idxs = [i for i in range(num_instances)]
    random.shuffle(idxs)
    train = []
    test = []
    bidx = int(np.ceil(num_instances*train_frac))

    train_idxs = idxs[:bidx]
    test_idxs = idxs[bidx:]
    return train_idxs, test_idxs
"""

def pred_to_one_hot(preds, class_labels):
    one_hots = []
    for pred in preds:
        one_hot = np.zeros(len(class_labels))
        one_hot[pred] = 1.0
        one_hots.append(one_hot)
    return np.array(one_hots)


def count(target, l):
    return len([i for i in l if i == target])

def downsample(true_labels, RANDOM_SEED):

    # downsample the data, such that the number of positive and negative examples is the same

    pos = count(0, true_labels)
    neg = count(1, true_labels)
    neg_idx = [i for i in range(len(true_labels)) if true_labels[i] == 1]
    pos_idx = [i for i in range(len(true_labels)) if true_labels[i] == 0]
    random.seed(RANDOM_SEED)
    if pos > neg:
        random.shuffle(pos_idx)
        pos_idx = pos_idx[:neg]

    elif neg > pos:
        random.shuffle(neg_idx)
        neg_idx = neg_idx[:pos]
    idx = neg_idx + pos_idx
    return idx

def upsample(true_labels, RANDOM_SEED):

    # upsample the data, such that the number of positive and negative examples is the same

    pos = count(0, true_labels)
    neg = count(1, true_labels)
    neg_idx = [i for i in range(len(true_labels)) if true_labels[i] == 1]
    pos_idx = [i for i in range(len(true_labels)) if true_labels[i] == 0]
    random.seed(RANDOM_SEED)

    sampled = []
    if pos > neg:
        while len(sampled) < len(pos_idx) - len(neg_idx):
            sampled += random.sample(neg_idx, np.min((pos-neg, neg)))
        neg_idx += sampled
    elif neg > pos:
        while len(sampled) < len(neg_idx) - len(pos_idx):
            sampled += random.sample(pos_idx, np.min((neg - pos, pos)))
        pos_idx += sampled

    idx = neg_idx + pos_idx
    return idx

def select(l, idx):
    return [l[i] for i in idx]



def load_json(fName):
    with codecs.open(fName, 'r', 'utf-8') as f:
        j = json.load(f)
    return j

def write_json(data, fname):
    with codecs.open(fname, 'w', 'utf-8') as f:
        json.dump(data, f)
    f.close()


def get_exp_path(config, params):
    """
    Create a directory to store the experiment.
    """
    # create the main dump path if it does not exist
    exp_folder = config.get('Dirs', 'resultdir')
    if not os.path.exists(exp_folder):
        subprocess.Popen("mkdir %s" % exp_folder, shell=True).wait()
    if not os.path.exists(exp_folder):
        subprocess.Popen("mkdir %s" % exp_folder, shell=True).wait()
    if params.exp_id == '':
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
        while True:
            exp_id = ''.join(random.choice(chars) for _ in range(10))
            exp_path = os.path.join(exp_folder, exp_id)
            if not os.path.isdir(exp_path):
                break
    else:
        exp_path = os.path.join(exp_folder, params.exp_id)
        assert not os.path.isdir(exp_path), exp_path
    # create the dump folder
    if not os.path.isdir(exp_path):
        subprocess.Popen("mkdir %s" % exp_path, shell=True).wait()
    return exp_path













