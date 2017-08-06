# -*- coding:utf-8 -*-
import copy, numpy as np
import pickle_use as pu
import time

# input variables
starting_lr = 0.025
lr=starting_lr
input_dim = 100
hidden_dim = 100
epoches_to_train = 1

# compute sigmoid nonlinerity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid result to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1-output)

def split(line):
    length = len(line)
    index=-1
    for i in range(length):
        if(line[i]==' '):
            index=i
            break
    assert index>-1, "must exist whitespace"
    return line[0:index], line[index+1:length-1]

# read data from process result of word2vec
def read_word2vec_data(word2vec_data_path):
    
    f=open(word2vec_data_path)
    lines = f.readlines()
    f.close()

    a=0
    vocab_size = int(lines[a])
    total_lines = len(lines)

    assert total_lines == 2*vocab_size+1, "total lines equal 2*vocab_size+1"

    process=1
    dictionary={}
    a=1
    for i in range(vocab_size):
        word, word_id = split(lines[a+i])
        dictionary[word]=int(word_id)
        if((a+i)*100.0/total_lines>process):
            process+=1
            print "read word2vec data:%%%d"%process

    a=vocab_size+1;
    word_vectors=[]
    for i in range(vocab_size):
        word_vector=[]
        wi=lines[a+i].split()
        for wij in wi:
            word_vector.append(float(wij))
        word_vectors.append(np.array([word_vector]))
        
        if((a+i)*100.0/total_lines>process):
            process+=1
            print "read word2vec data:%%%d"%process

    # 检测所有词向量，确保维度相同
    input_dim = len(word_vectors[0][0])
    for i in range(vocab_size):
        assert len(word_vectors[i][0]) == input_dim, "all vector should be the same size!"

    print "read word2vec data success!"
    print "dictionary size:%d, vector size:%d, input_dim:%d"%(len(dictionary), len(word_vectors), input_dim)
    return dictionary, word_vectors, input_dim

# read train data and convert to id(string to int) and store in datas 
def read_train_data(train_data, train_label, dictionary):
    datas=[]
    labels=[]
    
    f=open(train_data)
    data_lines = f.readlines()
    f.close()

    f=open(train_label)
    label_lines = f.readlines()
    f.close()

    assert len(data_lines)==len(label_lines), 'error,data num not equal label num'

    data_num = len(data_lines)

    process=1
    for i in range(data_num):
        if(i*100.0/data_num>process):
            process+=1
            print 'read train data:%%%d'%process

        # get the label of one instance
        label = int(label_lines[i])

        if(label==0):# unsuspend data, skip
            continue;
        if(label>=7):
            label=1
        else:
            label=0

        labels.append(label)
        
        # get the data of one instance
        sentence= data_lines[i].split()
        data=[]
        for word in sentence:
            if(dictionary.has_key(word)): # vocab
                index = dictionary[word]
            else:
                index=0
            data.append(index)
        datas.append(data)    

    
    print "read train data success!"
    print 'data size:%d, lable size:%d'%(len(datas), len(labels))
    return datas,labels
        
class RNNUnit():
    def __init__(self, word_vectors, sentences, labels):

        self.word_vectors = word_vectors
        self.sentences = sentences
        self.labels = labels
        # initialize recurrent neural network weights
        self.W_xh = 2*np.random.random((input_dim, hidden_dim)) - 1 # input and hidden weights
        self.W_hh = 2*np.random.random((hidden_dim, hidden_dim)) - 1 # hidden and hidden weights
        self.W_1h = 2*np.random.random((1, hidden_dim)) - 1 # classifier weights

    def train_model(self):
        total_times=0
        cross_entropy = 0
        sentence_num = len(self.sentences)
        start = time.time()
        for _ in range(epoches_to_train):
            for i in range(sentence_num):
                total_times+=1
                sentence = self.sentences[i]
                label = self.labels[i]
                cross_entropy+=self.train_one_sequence(sentence, label)
                if(total_times%250==0):
                    end=time.time()
                    print 'cross entropy:%f, percent:%%%f,time:%fs,speed: %fdata/s'%(cross_entropy/total_times, total_times*100.0/(sentence_num*epoches_to_train), end-start, 25/(end-start))
                    start=end
                    lr = starting_lr*(1-total_times*1.0/(epoches_to_train*sentence_num+1))
                    if(lr<starting_lr*0.0001):
                        lr=starting_lr*0.0001
                    
                    

    # train one instance of sequence
    def train_one_sequence(self, sentence, label):

        # store the update value of matrix
        W_xh_update = np.zeros_like(self.W_xh)
        W_hh_update = np.zeros_like(self.W_hh)
        W_1h_update = np.zeros_like(self.W_1h)
        
        length = len(sentence)
        
        # store previous layer 1 output of rnn unit as i-1 times
        hi_pre = np.zeros((1, hidden_dim))

        # moving along the word in sentence
        for word in sentence:

            # generate input and output
            xi = self.word_vectors[word]

            # hidden layer (input + prev_hidden)
            hi = sigmoid(np.dot(xi, self.W_xh) +
                              np.dot(hi_pre, self.W_hh))

            W_1h_update += hi
            W_xh_update += xi.T.dot(self.W_1h * sigmoid_output_to_derivative(hi))
            W_hh_update += hi_pre.T.dot(self.W_1h * sigmoid_output_to_derivative(hi))

            hi_pre = hi

        # generate classifier input and output
        x_input = np.zeros((1, hidden_dim)) # store input for classifier
        x_input = W_1h_update/length
        y_pre = sigmoid(x_input.dot(self.W_1h.T))
        y_pre = y_pre[0][0]
        y = label

        self.W_1h += lr * (y-y_pre)/length * W_1h_update
        self.W_xh += lr * (y-y_pre)/length * W_xh_update
        self.W_hh += lr * (y-y_pre)/length * W_hh_update

        # computer cross-entropy loss and return for show the change of gradient
        cross_entropy_loss = -1 * (y*np.log(y_pre) + (1-y)*np.log(1-y_pre))

        return cross_entropy_loss

    # predicts sentence sentiment
    def predicts(self, sentences, labels, log_path):
        assert len(sentences)==len(labels), "sentence num must equal label num"
        length = len(sentences)

        fwrite = open(log_path, 'w+')
        error=0.0;
        total_times=0
        error_num=20
        # 计算各个误差下的正确率
        for _ in range(error_num):
            error+=0.01
            predict_num = 0
            for i in range(length):
                total_times+=1
                if(total_times%250==0):
                    print "current predict percent:%lf"%(total_times*100.0/(error_num*length))
                sentence = sentences[i]
                label = labels[i]
                y=label

                # store previous layer 1 output of rnn unit as i-1 times
                hi_pre = np.zeros((1, hidden_dim))
                x_input = np.zeros((1, hidden_dim)) # store input for classifier

                # moving along the word in sentence
                for word in sentence:

                    # generate input and output
                    xi = self.word_vectors[word]

                    # hidden layer (input + prev_hidden)
                    hi = sigmoid(np.dot(xi, self.W_xh) +
                                      np.dot(hi_pre, self.W_hh))

                    x_input += hi
                    hi_pre = hi

                # generate classifier input and output

                x_input = x_input/length
                y_pre = sigmoid(x_input.dot(self.W_1h.T))

                if(y==1 and y-y_pre<error):
                    predict_num+=1
                elif(y==0 and y_pre-y<error):
                    predict_num+=1
                
            fwrite.write("error:%f, predict percent:%f\n"%(error, predict_num*100.0/length))
        fwrite.close()
        
        
    # load model from pickle file
    def import_model(self, model_path):
        model = pu.load_pickle(model_path)
        self.W_xh = model['W_xh']
        self.W_hh = model['W_hh']
        self.W_1h = model['W_1h']
        hidden_dim = model['hidden_dim']

    # export model data to pickle file
    def export_model(self, model_path):
        data = {
            'W_xh':self.W_xh,
            'W_hh':self.W_hh,
            'W_1h':self.W_1h,
            'hidden_dim':hidden_dim,
            'input_dim':input_dim
        }
        pu.save_pickle(data, model_path)



if __name__ == '__main__':
    # negative <=4, positive >=7
    word2vec_data_path = './bin/word2vec.data'
    train_data = './data/aclImdb/train-merge/data/train.data'
    train_label = './data/aclImdb/train-merge/label/train.label'
    
    dictionary, word_vectors, input_dim = read_word2vec_data(word2vec_data_path)

    datas,labels = read_train_data(train_data, train_label, dictionary)
    
    rnn_unit = RNNUnit(word_vectors, datas, labels)
    rnn_unit.train_model()
    
    rnn_model_path = './bin/rnn_unit.pickle'
    rnn_unit.export_model(rnn_model_path)
   # rnn_unit.import_model(rnn_model_path)

    test_data = './data/aclImdb/test-merge/data/train.data'
    test_label = './data/aclImdb/test-merge/label/train.label'
    log_path = "./bin/predict.log"
    test_datas,test_labels = read_train_data(test_data, test_label, dictionary)
    rnn_unit.predicts(test_datas, labels, log_path)
    
    
