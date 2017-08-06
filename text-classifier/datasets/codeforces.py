
from __future__ import absolute_import
from six.moves import zip
import numpy as np
import json
import warnings


def split_by_cate(xs, ys):
    pos=[]
    neg=[]
    for i in range(len(xs)):
        if ys[i]==1:
            pos.append(xs[i])
        elif ys[i]==0:
            neg.append(xs[i])
        else:
            raise ValueError('y cannt be value of ' +
                         str(ys[i]) + ', only 0 or 1.')
    return pos, neg

def get_sample(data, sample_num):
    
    xs=[]
    sample_index = np.arange(len(data))
    np.random.shuffle(sample_index)
    for i in range(sample_num):
        xs.append(data[sample_index[i]])
    return xs

def merge(pos_sample, neg_sample):
    merge_data=[]
    merge_label=[]
    assert(len(pos_sample)==len(neg_sample))
    for i in range(len(pos_sample)):
        merge_data.append(pos_sample[i])
        merge_data.append(neg_sample[i])
        merge_label.append(1)
        merge_label.append(0)
        
    return merge_data, merge_label
    
def get_balanced_data(xs, ys):
    pos, neg = split_by_cate(xs, ys)
        
    sample_num = len(neg)
    pos_sample = get_sample(pos, sample_num)
    neg_sample = get_sample(neg, sample_num)
    
    train_num = int(sample_num*0.5)
    train_data,train_label = merge(pos_sample[0:train_num], neg_sample[0:train_num])
    test_data,test_label = merge(pos_sample[train_num:], neg_sample[train_num:])
    
    return train_data,train_label,test_data,test_label
    
# load model data from file
def load_pickle(path):
    import pickle
    file = open(path, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def load_data(path=None, num_words=None, skip_top=0,
              maxlen=None, seed=113,
              start_char=1, oov_char=2, index_from=3, **kwargs):
    """Loads the CodeForces dataset.
    # Arguments
        path: where to cache the data (relative to `~/.keras/dataset`).
        num_words: max number of words to include. Words are ranked
            by how often they occur (in the training set) and only
            the most frequent words are kept
        skip_top: skip the top N most frequently occuring words
            (which may not be informative).
        maxlen: truncate sequences after this length.
        seed: random seed for sample shuffling.
        start_char: The start of a sequence will be marked with this character.
            Set to 1 because 0 is usually the padding character.
        oov_char: words that were cut out because of the `num_words`
            or `skip_top` limit will be replaced with this character.
        index_from: index actual words with this index and higher.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    # Raises
        ValueError: in case `maxlen` is so low
            that no input sequence could be kept.
    Note that the 'out of vocabulary' character is only used for
    words that were present in the training set but are not included
    because they're not making the `num_words` cut here.
    Words that were not seen in the training set but are in the test set
    have simply been skipped.
    """
    # Legacy support
    if 'nb_words' in kwargs:
        warnings.warn('The `nb_words` argument in `load_data` '
                      'has been renamed `num_words`.')
        num_words = kwargs.pop('nb_words')
    if kwargs:
        raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

    if path is None:
        path = '/home/lupeng/neural-network/data/codeforces_full.pkl'
    f = load_pickle(path)
    xs = f['datas']
    ys = f['labels']
    
    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if maxlen:
        new_xs = []
        new_ys = []
        for x, y in zip(xs, ys):
            if len(x) < maxlen:
                new_xs.append(x)
                new_ys.append(y)
        xs = new_xs
        ys = new_ys
    if not xs:
        raise ValueError('After filtering for sequences shorter than maxlen=' +
                         str(maxlen) + ', no sequence was kept. '
                         'Increase maxlen.')
    if not num_words:
        num_words = max([max(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [[oov_char if (w >= num_words or w < skip_top) else w for w in x] for x in xs]
    else:
        new_xs = []
        for x in xs:
            nx = []
            for w in x:
                if w >= num_words or w < skip_top:
                    nx.append(w)
            new_xs.append(nx)
        xs = new_xs

    train_data,train_label,test_data,test_label = get_balanced_data(xs, ys)

    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_label)
    
    np.random.seed(2*seed)
    np.random.shuffle(test_data)
    np.random.seed(2*seed)
    np.random.shuffle(test_label)
    
    
    x_train = np.array(train_data)
    y_train = np.array(train_label)

    x_test = np.array(test_data)
    y_test = np.array(test_label)

    return (x_train, y_train), (x_test, y_test)
