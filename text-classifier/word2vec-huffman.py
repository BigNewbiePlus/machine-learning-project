# -*- coding:utf-8 -*-
# 基于CBOW和Skip-gram模型，使用huffman树或negative sample构建词向量
# 本文使用CBOW+huffman构建，当然，也可以把所有情况均实现

__author__ = 'Big Fang'

import os
import numpy as np
import collections
import pickle_use as pu

from sklearn import preprocessing

save_path = "./bin/" # path for save model data
train_data = './data/test' # train data
eval_data = None # eval data
embedding_size = 100 # word vector
learning_rate = 0.1 # learning rate
windows_size = 5 # windows size
epoches_to_train = 1500 # times to train whole corpus

# 读取数据
def read_data(train_data):
    """读取数据集，获取分词序列"""
    if not os.path.exists(train_data):
        print("there is no exist data!")
        return None, None
    sentences = []
    words = []
    labels = []
    for line in open(train_data, 'r'):
        line_words = line.split()
        sentences.append(line_words[0:-1])
        labels.append(line_words[-1])
        words.extend(line_words[0:-1])

    return sentences, words, labels
       
# 创建数据集
def build_dataset(words):
    """创建数据集"""
    total_words = len(words) * 1.0
    dictionary={}
    count = collections.Counter(words).most_common(len(words))
    for word, word_count in count:
        dictionary[word] = Word(word_count/total_words, None,
                                2*np.random.random([1, embedding_size])-1)
    
    return dictionary, count

class Word():
    def __init__(self, probablity, huffman, vector):
        self.probablity = probablity # word probablity
        self.huffman = huffman # huffman code
        self.vector = vector # word vector
 
class HuffmanTreeNode():
    """huffman tree node"""
    def __init__(self, value, probablity):
        self.probablity = probablity # 该结点概率
        self.left = None # 左节点，为错分类点，编码1
        self.right = None # 右节点，为正分类点，编码0

        self.value = value # 该节点所在的值，叶节点存放词，中间节点存放中间向量
        self.huffman = '' # 从根结点到当前节点huffman路径,初始化为空
        

# 创建huffman树
class HuffmanTree():
    def __init__(self, dictionary):
        """初始化huffman树"""
        self.root = None # root node

        huffman_leaf_nodes = [HuffmanTreeNode(word, attri.probablity) # huffman nodes build from dictioanry
                              for word, attri in dictionary.items()]

        self.build_tree(huffman_leaf_nodes) # create huffman tree
        self.generate_huffman_code(self.root, dictionary) # create huffman code

    def build_tree(self, huffman_leaf_nodes):
        """根据叶节点构建huffman树"""

        #获取叶节点
        huffman_nodes=huffman_leaf_nodes
        
        while len(huffman_nodes)>1:
            i1=0 # i1表示概率最小的节点
            i2=1 # i2表示概率第二小的节点

            if huffman_nodes[i2].probablity < huffman_nodes[i1].probablity:
                i1,i2 = i2,i1

            for i in range(2, len(huffman_nodes)): #遍历以找到最小概率两节点
                if huffman_nodes[i].probablity < huffman_nodes[i2].probablity:
                    i2=i
                    if huffman_nodes[i2].probablity<huffman_nodes[i1].probablity:
                        i1,i2 = i2,i1

            merge_node = self.merge(huffman_nodes[i1], huffman_nodes[i2])

            if i1<i2:
                huffman_nodes.pop(i2)
                huffman_nodes.pop(i1)
            elif i2<i1:
                huffman_nodes.pop(i1)
                huffman_nodes.pop(i2)
            else:
                raise RuntimeError('i1 index should not equal to i2')

            huffman_nodes.insert(0, merge_node)

        # 最后一个元素就是根节点
        self.root = huffman_nodes[0]

    def generate_huffman_code(self, root, dictionary):
        stack = [root] # 堆栈方式遍历，为了遍历根结点
        while len(stack)>0: # open表存在未遍历的
            s = stack.pop()
            while s.left and s.right: # 二叉树，左右节点同时存在
                s.left.huffman = s.huffman+'1'
                s.right.huffman = s.huffman+'0'
                stack.append(s.right)
                s = s.left

            word = s.value # leaf node, value store the word
            huffman_code = s.huffman # get the huffman code

            dictionary[word].huffman = huffman_code # store the huffman code
        
    def merge(self, node1, node2):
        merge_node_probablity = node1.probablity + node2.probablity
        merge_node = HuffmanTreeNode(np.zeros([1, embedding_size]), merge_node_probablity)
        # 默认将概率大的左边分配, 当然也可以按照其他方式分配，如随机
        if node1.probablity >= node2.probablity:
            merge_node.left = node1
            merge_node.right = node2

        else:
            merge_node.left = node2
            merge_node.right = node1
        return merge_node

class Word2Vec():
    def __init__(self, dictionary, sentences, labels, model_type='cbow'):
        self.model_type = model_type
        self.dictionary = dictionary
        self.sentences = sentences
        self.labels = labels

    def train_model(self):
        skip_num = (windows_size-1)/2
        self.huffman_tree = HuffmanTree(self.dictionary)
        print 'create huffman tree success!'
        print '\ntrain begin:'
        total_times=0
        for epoch in range(epoches_to_train): # epoch to train
            for sentence in self.sentences: # sentence number to train
                length = len(sentence)
                for i in range(length):
                    total_times +=1
                    if total_times % 2000 == 0:
                        print 'train times %d'%total_times
                    self.cbow_model_train(sentence[i],
                                          sentence[max(0,i-skip_num):i] + sentence[i+1:min(i+1+skip_num,length)],
                                          self.dictionary)

    def cbow_model_train(self, word, context_words, dictionary):
        x = np.zeros([1, embedding_size]) # store sum of context words vector
        for context_word in context_words:
            x += dictionary[context_word].vector
        x = preprocessing.normalize(x) # normalized vector
        huffman = dictionary[word].huffman # gen the huffman of the word

        node = self.huffman_tree.root
        e = np.zeros([1, embedding_size])
        
        for i in range(len(huffman)): # iterate the huffman code
            huffman_bit = huffman[i] # huffman bit
            q = self.sigmoid(x.dot(node.value.T)) # classifier
            gradient = learning_rate * (1-int(huffman_bit)-q)
            e += gradient * node.value
            node.value += gradient * x # update intermediate vector of huffman node
            node.value = preprocessing.normalize(node.value)
            if huffman_bit == '0': # 0 means positive class and right
                node = node.right # 
            else:
                node = node.left # 1 means left and negative class

        # update vector of context word
        for context_word in context_words:
            dictionary[context_word].vector += e
            dictionary[context_word].vector = preprocessing.normalize(dictionary[context_word].vector)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    # load model from file
    def import_model(self, model_path):
        model = pu.load_pickle(model_path)
        self.dictionary = model['dictionary']
        self.sentences = model['sentences']
        self.labels = model['labels']
        embedding_size = model['embedding_size']

    def export_model(self, model_path):
        data = {
            'dictionary':self.dictionary,
            'sentences': self.sentences,
            'labels': self.labels,
            'embedding_size': embedding_size
        }
        pu.save_pickle(data, model_path)

def plot_with_labels(dictionary,count, plot_num, filename='tsne.png'):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

    low_dim_embs = []
    labels=[]
    i=0
    for word, _ in count:
        labels.append(word)
        low_dim_embs.append(dictionary[word].vector[0])  # vector:[1,embedding_size], low_dim_embs must be 2-D dimension
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
    sentences, words, labels = read_data(train_data=train_data)
    print 'read data success!'

    dictionary, count = build_dataset(words)
    del words # free unused memory
    print 'build dataset success!'
    
    word2vector = Word2Vec(dictionary, sentences, labels, model_type='cbow')
    word2vector.train_model()
    word2vector.export_model(save_path+'word2vec.pickle')

    word2vector.import_model(save_path + 'word2vec.pickle')

    for word,Word_ in word2vector.dictionary.items():
        print Word_.vector,Word_.probablity
    print word2vector.sentences
    print word2vector.labels
    print embedding_size
    plot_with_labels(dictionary, count, min(100,len(count)))
