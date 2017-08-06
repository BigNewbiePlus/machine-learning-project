/*
 * author : Big~Fang
 * model : rnn
 */

#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<map>
#include<stdlib.h>
#include<time.h>
#include<sys/time.h>
#include<pthread.h>
#include<math.h>
#include<atomic>
using namespace std;

const float starting_lr = 0.025;
int vocab_size = 0;
int input_dim = 100;
int hidden_dim = 200;
int epoches_to_train = 2;
float lr = starting_lr;
std::atomic<int> total_times(0);
const int num_threads = 10;

vector<float> sigmoid_lookup_table;
vector<float> logarithm_lookup_table;

vector<vector<int> > datas, test_datas;
vector<int> labels, test_labels;
float* word_vectors, *W_xh, *W_hh, *W_1h;
bool re_train=false;

class RNNUnitVars{
public:
  RNNUnitVars();
  ~RNNUnitVars();

  //store the update value of matrix
  float* W_xh_update;
  float* W_hh_update;
  float* W_1h_update;

  //store previous layer 1 output of rnn unit as i-1 times
  float* hi_pre;
  
  float* hi;
  float* h_temp1;
  float* h_temp2;
  float* xh_temp;
  float* hh_temp;

  //generate classifier input and output
  float* x_input;
};

RNNUnitVars::RNNUnitVars()
{
  W_xh_update = new float[input_dim*hidden_dim];
  W_hh_update = new float[hidden_dim*hidden_dim];
  W_1h_update = new float[hidden_dim];

  hi_pre = new float[hidden_dim];
  
  hi = new float[hidden_dim];

  h_temp1 = new float[hidden_dim];
  h_temp2 = new float[hidden_dim];
  
  xh_temp = new float[input_dim*hidden_dim];
  hh_temp = new float[hidden_dim*hidden_dim];

  x_input = new float[hidden_dim];
}

RNNUnitVars::~RNNUnitVars()
{
  delete[] W_1h_update;
  delete[] W_xh_update;
  delete[] W_hh_update;

  delete[] hi_pre;
  delete[] hi;
  delete[] h_temp1;
  delete[] h_temp2;
  delete[] xh_temp;
  delete[] hh_temp;

  delete[] x_input;
}

void random_init_vector(float* a, int length)
{
  for(int i=0;i<length;i++){
    a[i] = (rand()%2000-1000)*0.001;
  }
}

inline void zeros_vector(float* a, int length)
{
  for(int i=0;i<length;i++)
    a[i]=0;
}
inline void dot_vector(float* ac, float* ab, float* bc, int a, int b, int c)
{
  for(int row=0;row<a;row++){
    for(int col=0;col<c;col++){
      float f=0;
      for(int k=0;k<b;k++){
	f+=ab[row*b+k]*bc[k*c+col];	
      }
      ac[row*c+col]=f;
    }
  }
}

//sum elelment-wise
inline void sum_vector(float* a, float* b, int length)
{
  for(int i=0;i<length;i++)
    a[i] += b[i];
}

//sum elelment-wise
inline void sum_vector_by_scale(float*a, float*b, int length, float scale)
{
  for(int i=0;i<length;i++)
    a[i] += b[i]*scale;
}
// multiply element-wise
inline void mul_vector(float* a, float* b, int length)
{
  for(int i=0;i<length;i++){
    a[i] = a[i]*b[i];
  }
}
//inner product vector
inline float inn_vector(float* a, float* b, int length)
{
  float f=0;
  for(int i=0;i<length;i++)
    f+=a[i]*b[i];
  return f;
}

//copy element-wise
inline void cop_vector(float*a, float*b, int length)
{
  for(int i=0;i<length;i++)
    a[i]=b[i];
}

inline void print_vector(float*a, int row, int col)
{
  cout<<row<<" row,"<<col<<" col\n";
  for(int i=0;i<row;i++){
    for(int j=0;j<col;j++)
      cout<<a[i*col+j]<<",";
    cout<<"\n\n";
  }
}

inline float sigmoid(float x){
  if(x<-10)x=-10;
  if(x>10)x=10;
  int index = round((x+10)*100);
  return sigmoid_lookup_table[index];
}

inline void sig_vector(float* result, float* x, int length){
  for(int i=0;i<length;i++){
    result[i] = sigmoid(x[i]);
  }
}

inline float logarithm(float x)
{
  int index = round(x*1000);
  return logarithm_lookup_table[index];
}

inline void sig_der_vector(float* result, float* x, int length)
{
  for(int i=0;i<length;i++)
    result[i] = x[i]*(1-x[i]);
}

void gen_sigmoid_lookup_table()
{
  int K = 2000; //total seperate
  float step = 0.01;// #[-10, +10]
  
  float x, output;
  for(int i=0; i<=K;i++){
    x = -10 + i*step;
    output = 1/(1+exp(-x));
    sigmoid_lookup_table.push_back(output);
  }
}

void gen_logarithm_lookup_table()
{
  int K=1000;
  float step=0.001;

  float output;
  logarithm_lookup_table.push_back(log(0.0001));
  for(int i=1;i<=K;i++){
    output = log(i*step);
    logarithm_lookup_table.push_back(output);
  }
}

inline int get_input_dim(string &line){
  int length = line.size();
  int count=0;
  for(int i=0;i<length;i++){
    if(line[i]==' ')
      count++;
  }
  return count+1;
}

inline vector<string> split(string line)
{
  vector<string> sentence;
  string word;
  int length  = line.size();
  for(int i=0;i<length;i++){
    if(line[i]==' '){
      sentence.push_back(word);
      word.clear();
    }else
      word.push_back(line[i]);
  }
  sentence.push_back(word);

  return sentence;
}
//读取词向量
void read_word2vec_data(string word2vec_data_path, map<string,int> &dictionary, float* &word_vectors)
{
  ifstream is(word2vec_data_path.c_str());
  string line;
  stringstream ss;

  if(is){
    getline(is,line);
    ss<<line;
    ss>>vocab_size;

    int process=1; //记录数据处理进度
    
    string word;
    int id;
    for(int i=0;i<vocab_size;i++){
      getline(is,line);

      vector<string> splits = split(line);

      if(splits.size()!=2){
	cout<<"error,dictioanry!"<<endl;
	exit(0);
      }

      ss.clear();
      ss<<splits[1];
      ss>>id;
      dictionary[splits[0]]=id;
      if(id>vocab_size){
	cout<<"辞典id错误,不能大于辞典大小,"<<line<<endl;
	exit(0);
      }

      if((1+i)*100.0/(vocab_size*2+1)>process){
	process+=1;
	cout<<"read word2vec data:%"<<process<<endl;
      }
    }

    int temp_dim, a;
    float wij;
    for(int i=0;i<vocab_size;i++){
      getline(is,line);
      
      temp_dim = get_input_dim(line);
      if(i==0){
	input_dim = temp_dim;
	word_vectors = new float[vocab_size*input_dim];
      }
      else if(temp_dim!=input_dim){//校验词向量维度相同
	cout<<"error,word vector dim not same"<<endl;
	exit(0);
      }

      ss.clear();
      ss<<line;
      a=i*input_dim;
      for(int j=0;j<input_dim;j++){
	ss>>wij;
	word_vectors[a+j]=wij;
      }

      if((vocab_size+1+i)*100.0/(vocab_size*2+1)>process){
	process+=1;
	cout<<"read word2vec data:%"<<process<<endl;
      }
    }

    is.close();
    
    cout<<"read word2vec data success!"<<endl;
    cout<<"dictionary size:"<<dictionary.size()
	<<",vector size:"<<vocab_size
	<<",input_dim:"<<input_dim<<endl;
  }
  else{
    cout<<"error, no such file!"<<endl;
  }
}

//读取数据
void read_train_data(string train_data, string train_label,map<string,int> &dictionary,vector<vector<int> > &datas, vector<int> &labels)
{
  ifstream is_data(train_data.c_str());
  ifstream is_label(train_label.c_str());

  string line_data, line_label;

  int label;// 处理标签
  stringstream ss;

  int word_num, index; //处理数据
  map<string,int>::iterator it_map;

  int total_num=0;
  
  if(is_data && is_label){
    while(getline(is_data, line_data) && getline(is_label, line_label)){

      total_num++;
      if(total_num%1000==0)
	cout<<"read train data:"<<total_num/1000<<"k"<<endl;
      //先处理标签
      ss.clear();
      ss<<line_label;
      ss>>label;

      if(label==0) continue; //无监督数据，跳过
      else if(label>=7)label=1; //正例
      else label=0;//负例
      labels.push_back(label);

      //处理数据
      vector<int> data;
      vector<string> sentence = split(line_data);
      word_num = sentence.size();
      for(int i=0;i<word_num;i++){
	it_map = dictionary.find(sentence[i]);
	if(it_map==dictionary.end()) //未发现该单词
	  index=0;
	else
	  index = it_map->second; //id
	data.push_back(index);
      }
      datas.push_back(data);
    }
  }
  else if(!is_data)cout<<"data file can't open!"<<endl;
  else if(!is_label)cout<<"label data can't open!"<<endl;

  is_data.close();
  is_label.close();

  cout<<"read train data success!\n"<<"data size:"<<datas.size()<<",label size:"<<labels.size()<<endl;
}

float train_one_sequence(vector<int> &sentence, int label, float* word_vectors, float* W_xh, float* W_hh, float* W_1h, RNNUnitVars& rnn_unit_vars)
{
  zeros_vector(rnn_unit_vars.W_xh_update, input_dim*hidden_dim);
  zeros_vector(rnn_unit_vars.W_hh_update, hidden_dim*hidden_dim);
  zeros_vector(rnn_unit_vars.W_1h_update, hidden_dim);

  int length = sentence.size();

  zeros_vector(rnn_unit_vars.hi_pre, hidden_dim);
  
  float* xi;
  int word;
  
  for(int i=0;i<length;i++){
    
    word = sentence[i];

    xi = word_vectors+ word*input_dim; //获取词向量指针
    
    dot_vector(rnn_unit_vars.h_temp1, xi, W_xh,1,input_dim, hidden_dim);
   
    dot_vector(rnn_unit_vars.h_temp2,rnn_unit_vars.hi_pre, W_hh, 1, hidden_dim, hidden_dim);

    sum_vector(rnn_unit_vars.h_temp1, rnn_unit_vars.h_temp2, hidden_dim);
    sig_vector(rnn_unit_vars.hi, rnn_unit_vars.h_temp1, hidden_dim);
    sig_der_vector(rnn_unit_vars.h_temp1,rnn_unit_vars.hi, hidden_dim);
    
    mul_vector(rnn_unit_vars.h_temp1, W_1h, hidden_dim);

    dot_vector(rnn_unit_vars.xh_temp, xi, rnn_unit_vars.h_temp1,input_dim,1,hidden_dim);
    dot_vector(rnn_unit_vars.hh_temp, rnn_unit_vars.hi_pre, rnn_unit_vars.h_temp1,hidden_dim,1,hidden_dim);

    sum_vector(rnn_unit_vars.W_1h_update, rnn_unit_vars.hi, hidden_dim);
    sum_vector(rnn_unit_vars.W_xh_update, rnn_unit_vars.xh_temp, input_dim*hidden_dim);
    sum_vector(rnn_unit_vars.W_hh_update, rnn_unit_vars.hh_temp, hidden_dim*hidden_dim);
    cop_vector(rnn_unit_vars.hi_pre, rnn_unit_vars.hi, hidden_dim);
  }
  
  zeros_vector(rnn_unit_vars.x_input, hidden_dim);

  sum_vector_by_scale(rnn_unit_vars.x_input, rnn_unit_vars.W_1h_update,hidden_dim, 1.0/length);
  
  float y_pre = sigmoid(inn_vector(rnn_unit_vars.x_input, W_1h,hidden_dim));
  float y=label;

  float scale = lr*(y-y_pre)/length;

  sum_vector_by_scale(W_1h, rnn_unit_vars.W_1h_update,hidden_dim, scale);
  sum_vector_by_scale(W_xh, rnn_unit_vars.W_xh_update,input_dim*hidden_dim, scale);
  sum_vector_by_scale(W_hh, rnn_unit_vars.W_hh_update,hidden_dim*hidden_dim, scale);

  //计算交叉熵
  float cross_entropy = -y*logarithm(y_pre) - (1-y)*logarithm(1-y_pre);
 
  return cross_entropy;
}

void *rnn_train_model_thread(void *id)
{
  time_t start, end;
  struct timeval tstart,tend;
  gettimeofday(&tstart,NULL);
  double cross_entropy = 0;

  long thread_id = long(id);
  int data_num = datas.size();
  int thread_data_num = data_num/num_threads;
  int a = thread_id*thread_data_num;
  int b = a+thread_data_num;

  int block=150;
  int last_total_times = 0;
  int one_total_times=0;
  RNNUnitVars rnn_unit_vars;

  
  start = clock();
  ofstream os("./bin/rnn_cross_entropy.change");
  for(int epoch=0;epoch<epoches_to_train;epoch++)
    for(int i=a;i<b;i++){
      total_times+=1;
      one_total_times+=1;
      vector<int> &sentence = datas[i];
      int label = labels[i];

      cross_entropy+=train_one_sequence(sentence, label, word_vectors, W_xh, W_hh, W_1h, rnn_unit_vars);

      if((thread_id == 0) && (one_total_times%block==0)){
        gettimeofday(&tend,NULL);
        long total_time = tend.tv_sec - tstart.tv_sec;
	cout<<"thread:"<<thread_id
	    <<",cross entropy:"<<cross_entropy/one_total_times
	    <<",percent:%"<<one_total_times*100.0/((b-a)*epoches_to_train)
	    <<",time:"<<total_time
	    //<<"s,speed:"<<(total_times-last_total_times)*CLOCKS_PER_SEC/(end-start)*num_threads
	    <<"s,speed:"<<(total_times-last_total_times)/(total_time+0.00001)
	    <<"sen/s"<<endl;
	os<<"thread:"<<thread_id
	    <<",cross entropy:"<<cross_entropy/one_total_times
	    <<",percent:%"<<one_total_times*100.0/((b-a)*epoches_to_train)
	    <<",time:"<<total_time
	    //<<"s,speed:"<<(total_times-last_total_times)*CLOCKS_PER_SEC/(end-start)*num_threads
	    <<"s,speed:"<<(total_times-last_total_times)/(total_time+0.00001)
	    <<"sen/s"<<endl;
	start=end;
	tstart.tv_sec = tend.tv_sec;
	last_total_times=total_times;
	lr = starting_lr*(1-total_times*1.0/(epoches_to_train*data_num+1));
	if(lr<starting_lr*0.0001)
	  lr = starting_lr*0.0001;
      }
    }
  os.close();
}

float predict_one_sequence(vector<int> &sentence, float* word_vectors, float* W_xh, float* W_hh, float* W_1h)
{
  int length = sentence.size();
  
  float* xi;
  int word;

  float* hi = new float[hidden_dim];
  float* hi_pre = new float[hidden_dim];
  float* x_input = new float[hidden_dim];

  float* temp1 = new float[hidden_dim];
  float* temp2 = new float[hidden_dim];

  zeros_vector(hi_pre, hidden_dim);
  zeros_vector(x_input, hidden_dim);

  for(int i=0;i<length;i++){
    
    word = sentence[i];

    xi = word_vectors+ word*input_dim; //获取词向量指针
    
    dot_vector(temp1, xi, W_xh,1,input_dim, hidden_dim);
   
    dot_vector(temp2, hi_pre, W_hh, 1, hidden_dim, hidden_dim);
    
     sum_vector(temp1, temp2, hidden_dim);
     
     sig_vector(hi,temp1, hidden_dim);

    sum_vector(x_input, hi, hidden_dim);

    cop_vector(hi_pre, hi, hidden_dim);
  }

  for(int i=0;i<hidden_dim;i++)
    x_input[i]/=length;
  
  float y_pre = sigmoid(inn_vector(x_input, W_1h, hidden_dim));

  delete[] hi;
  delete[] hi_pre;
  delete[] x_input;
  delete[] temp1;
  delete[] temp2;
  
  return y_pre;
}

void predicts(vector<vector<int> > &test_datas, vector<int> &labels, string log_path, float* word_vectors, float* W_xh, float* W_hh, float* W_1h)
{
  int datas_num = test_datas.size();
  int right_num;

  int total_num=0;
  int block = 250;
  vector<float> y_pres;
  for(int i=0;i<datas_num;i++){
    vector<int> &data = test_datas[i];

    total_num++;
    if(total_num%block==0)
      cout<<"predict percent:%"<<total_num*100.0/datas_num<<endl;

	
    float y_pre = predict_one_sequence(data, word_vectors, W_xh, W_hh, W_1h);
    y_pres.push_back(y_pre);
  }
  
  if(datas_num!=y_pres.size()){
    cout<<"error!"<<endl;
    exit(0);
  }

  ofstream fwrite(log_path.c_str());
  if(!fwrite.is_open()){
    cout<<"can't open log file:"<<log_path<<endl;
    exit(0);
  }
  
  int y;
  float y_pre;
  for(float error=0.01;error<=0.021;error+=0.01){
    right_num=0;
    for(int i=0;i<datas_num;i++){
      y = labels[i];
      y_pre=y_pres[i];
      if(((y==1) && (y-y_pre<=error))||((y==0) &&(y_pre<=error))){
	right_num++;
	fwrite<<"right:"<<y<<"-"<<y_pre<<endl;
      }else
	fwrite<<"wrong:"<<y<<"-"<<y_pre<<endl;
    }
     cout<<"error:"<<error<<",percent:"<<right_num*100.0/datas_num<<endl;
    fwrite<<"error:"<<error<<",percent:"<<right_num*100.0/datas_num<<endl;
  }
  fwrite.close();
}

void init_rnn_net()
{
  W_xh = new float[input_dim*hidden_dim];
  W_hh = new float[hidden_dim*hidden_dim];
  W_1h = new float[1*hidden_dim];

  //initialize recurrent neural network weights
  random_init_vector(W_xh, input_dim*hidden_dim);
  random_init_vector(W_hh, hidden_dim*hidden_dim);
  random_init_vector(W_1h, 1*hidden_dim);
}

void save_rnn_unit_data(float* W_xh, float* W_hh, float* W_1h, std::string path)
{
  ofstream os(path.c_str());
  if(os){
    int length = input_dim*hidden_dim;
    os<<length<<endl;
    for(int i=0;i<length-1;i++)os<<W_xh[i]<<" ";os<<W_xh[length-1]<<endl;

    length=hidden_dim*hidden_dim;
    os<<length<<endl;
    for(int i=0;i<length-1;i++)os<<W_hh[i]<<" ";os<<W_hh[length-1]<<endl;

    length=1*hidden_dim;
    os<<length<<endl;
    for(int i=0;i<length-1;i++)os<<W_1h[i]<<" ";os<<W_1h[length-1]<<endl;
    cout<<"savb rnn data success!"<<endl;
  }
}

void read_rnn_unit_data(float* W_xh, float* W_hh, float* W_1h, std::string path)
{
  ifstream is(path.c_str());
  string line;
  int length;
  float wij;
  stringstream ss;

  if(is){
    getline(is,line);ss.clear();ss<<line;ss>>length;
    getline(is,line);ss.clear();ss<<line;
    for(int i=0;i<length;i++){ss>>wij;W_xh[i]=wij; }

    getline(is,line);ss.clear();ss<<line;ss>>length;
    getline(is,line);ss.clear();ss<<line;
    for(int i=0;i<length;i++){
      ss>>wij;
      W_hh[i]=wij;
    }

    getline(is,line);ss.clear();ss<<line;ss>>length;
    getline(is,line);ss.clear();ss<<line;
    for(int i=0;i<length;i++){
      ss>>wij;
      W_1h[i]=wij;
    }
    cout<<"read data success!"<<endl;
  }else{
    cout<<"can't read rnn unit data"<<endl;
    exit(0);
  }

  is.close();
}
int main()
{
  string word2vec_data_path = "./bin/word2vec.data";
  string train_data = "./data/aclImdb/train-merge/data/train.data";
  string train_label = "./data/aclImdb/train-merge/label/train.label";

  map<string, int> dictionary;

  read_word2vec_data(word2vec_data_path, dictionary, word_vectors);
  
  read_train_data(train_data, train_label, dictionary, datas, labels);

  gen_sigmoid_lookup_table();
  gen_logarithm_lookup_table();
  cout<<"gen sigmoid and logarithm lookup success!"<<endl;

  init_rnn_net();

  string rnn_data_save_path = "./bin/rnn.data";

  if(re_train){
    pthread_t *pt = new pthread_t[num_threads];
    long a;
    for(a=0;a<num_threads;a++)pthread_create(&pt[a], NULL, rnn_train_model_thread,(void*)a);
    for(a=0;a<num_threads;a++)pthread_join(pt[a], NULL);
    save_rnn_unit_data(W_xh, W_hh, W_1h, rnn_data_save_path);
  }else{
    read_rnn_unit_data(W_xh, W_hh, W_1h, rnn_data_save_path);
  }

   string test_data = "./data/aclImdb/test-merge/data/train.data";
   string test_label = "./data/aclImdb/test-merge/label/train.label";
   string log_path = "./bin/predict.log";

   read_train_data(test_data, test_label, dictionary, test_datas, test_labels);
   predicts(test_datas, labels, log_path, word_vectors, W_xh, W_hh, W_1h);

  delete[] W_xh;
  delete[] W_hh;
  delete[] W_1h;
}
