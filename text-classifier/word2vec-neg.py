# -*- coding:utf-8 -*-
# model build based on CBOW + Negative Sampling

__author__ = 'Big Fang'

import os
import numpy as np
import collections
import pickle_use as pu
import time
import copy

from sklearn import preprocessing

data_type = 'text8' # imdb, text8
save_path = "./bin/" # path for save model data
eval_data = None # eval data
embedding_size = 100 # word vector
learning_rate = 0.1 # learning rate
windows_size = 5 # windows size
min_count = 5 # the minimum number of word occurrence for it to be included in the vocabulary
num_sampled = 32 # Number of negative examples to sample
epoches_to_train = 1 # times to train whole corpus

# 读取数据
def read_text8_data(train_data):
    """读取数据集，获取分词序列"""
    if not os.path.exists(train_data):
        print("there is no exist data!")
        return None, None
    sentences = []
    words = []
    for line in open(train_data, 'r'):
        sentences.append(line.split())
        words.extend(line.split())
    return sentences, words

# read IMDB data
def read_imdb_data(imdb_data_path):
    """read IMDB data"""
    if not os.path.exists(imdb_data_path):
        print('there is no exist imdb direcory!')
        return None, None
    
    sentences = []
    words = []

    # get all file from imdb directory
    for parent, dirnames, filenames in os.walk(imdb_data_path):
        for filename in filenames:
            filepath = os.path.join(parent, filename)
            for line in open(filepath, 'r'):
                sentences.append(line.split())
                words.extend(line.split())
    
    return sentences, words
       
# build dataset, words occurence with at least min_count as vocabulary
def build_dataset(sentences, words):
    """创建数据集"""

    # build count data
    count = [['UNK', -1]]
    most_commons = collections.Counter(words).most_common(len(words))

    i=0
    for word, word_count in most_commons:
        i+=1
        if word_count<min_count:
            break
    count.extend(most_commons[0:i])
    print 'total vocabulary size change: %d->%d'%(len(most_commons), len(count))

    # build dictionary
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    # build datas
    datas = []
    unk_count=0
    for sentence in sentences:
        data=[]
        for word in sentence:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0 # dictionary['UNK']
                unk_count += 1
            data.append(index)
        datas.append(copy.deepcopy(data))
        
    count[0][1] = unk_count
    
    # build reverse dictionary
    rev_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    # build word vector and intermediate vector for cbow and negative sampling model
    vocab_size = len(count)
    word_vectors = []
    inte_vectors = []

    for i in range(vocab_size):
        word_vectors.append(2*np.random.random([embedding_size])-1)
        inte_vectors.append(2*np.random.random([embedding_size])-1)
    
    return count, dictionary, rev_dictionary, datas, word_vectors, inte_vectors


class Word2Vec():
    def __init__(self,dictioary, vocab_size, datas, word_vectors, inte_vectors, model_type='cbow'):
        self.model_type = model_type
        self.dictionary = dictionary
        self.vocab_size = vocab_size
        self.datas = datas
        self.word_vectors = word_vectors
        self.inte_vectors = inte_vectors

    def train_model(self):
        skip_num = (windows_size-1)/2
        
        total_times = 0
        cross_entropy_loss = 0
        
        for epoch in range(epoches_to_train): # epoch to train
            for data in self.datas: # sentence number to train
                        
                length = len(data)
                neg_words = self.negative_sampling() # negative sampling words, one sentence use the same sampling
                for i in range(length):
                    word = data[i] # word
                    context_words = data[max(0,i-skip_num):i] + data[i+1:min(i+1+skip_num,length)] # context words

                    loss = self.cbow_model_train(word, context_words, neg_words) # train one instance

                    cross_entropy_loss += loss
                    total_times +=1

                    if total_times % 10000 == 0:
                        print 'train times: %d, average cross_entropy_loss: %lf.'%(total_times, cross_entropy_loss/10000)
                        cross_entropy_loss = 0
                    

    def cbow_model_train(self, word, context_words, neg_words):

        # compute cross-entropy-loss
        cross_entropy_loss = 0
        
        # store sum of context words vector
        x_w = np.zeros([embedding_size])
        
        for context_word in context_words:
            x_w += self.word_vectors[context_word]
        # update for context word vector

        e = np.zeros([embedding_size])

        # negative words iterate
        for neg_word in neg_words: # iterate the negative words
            theta_u = self.inte_vectors[neg_word]
            
            q = self.sigmoid(x_w.dot(theta_u.T)) # classifier, dim 1, 1
            g = learning_rate * (-q) # gradient
            e += g*theta_u
            theta_u += g*x_w

            # update intermediate vector for negative sample word
            self.inte_vectors[neg_word] = theta_u

            # loss compute
            cross_entropy_loss += -np.log(1-q)

        
        # positive word iterate(only one)
        theta_u = self.inte_vectors[word]

        q = self.sigmoid(x_w.dot(theta_u.T)) # classifier
        g = learning_rate * (1-q) # q dim
        e += g*theta_u
        theta_u += g*x_w
        self.inte_vectors[word] = theta_u

        cross_entropy_loss += -np.log(q)
        
        # update vector of context word
        for context_word in context_words:
            self.word_vectors[context_word] += e

        return cross_entropy_loss

    # negative sampling for producing negative class
    def negative_sampling(self):
        neg_words = np.random.randint(self.vocab_size, size=num_sampled)
        return neg_words

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    # load model from file
    def import_model(self, model_path):
        model = pu.load_pickle(model_path)
        self.dictionary = model['dictionary']
        self.word_vectors = model['word_vectors']
        embedding_size = model['embedding_size']

    def export_model(self, model_path):
        # only save word vector and embedding_size
            
        data = {
            'dictionary':self.dictionary,
            'word_vectors': self.word_vectors,
            'embedding_size': embedding_size
        }
        pu.save_pickle(data, model_path)

def plot_with_labels(dictionary, word_vectors, count, plot_num, filename='tsne.png'):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

    low_dim_embs = []
    labels=[]
    i=0

    # store the most frequency words of first plot_num
    for word, _ in count:
        word_id = dictionary[word]
        
        labels.append(word)
        low_dim_embs.append(word_vectors[word_id])  # vector:[1,embedding_size], low_dim_embs must be 2-D dimension
        i+=1
        if i>plot_num:
            break

    low_dim_embs = tsne.fit_transform(low_dim_embs)
    
    plt.figure(figsize=(18, 18)) # in inches
    for i, label in enumerate(labels):
        x,y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x,y),
                     xytext=(5,2),
                     textcoords = 'offset points',
                     ha='right',
                     va='bottom')

        plt.savefig(filename)
        
if __name__ == '__main__':

    # read data based on data type
    if data_type == 'text8':
        text8_data_path = './data/text8'
        sentences, words = read_text8_data(text8_data_path)
    elif data_type == 'imdb':
        imdb_data_path = './data/aclImdb/train'
        sentences, words = read_imdb_data(imdb_data_path)

    print 'read text data success! sentence num: %d, word num: %d.'%(len(sentences), len(words))
    
    count, dictionary, rev_dictionary, datas, word_vectors, inte_vectors = build_dataset(sentences, words)

    print 'build dataset success! vocab_size: %d'%(len(count))
    print 'Most 5 frequency words:',count[0:5]

    # free unused memory
    del sentences
    del words

    
    vocab_size = len(count)
    # initialize word2vector model
    word2vector = Word2Vec(dictionary, vocab_size, datas, word_vectors, inte_vectors, model_type='cbow')

    # train
    word2vector.train_model()

    # export model data
    word2vector.export_model(save_path+'word2vec.pickle')

    # import model data
   # word2vector.import_model(save_path + 'word2vec.pickle')

    # plot the most frequency 500 words
    plot_with_labels(dictionary, word_vectors, count, min(500, vocab_size))
