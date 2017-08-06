#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<sys/stat.h>
#include<unistd.h>
#include<dirent.h>
#include<stdlib.h>

using namespace std;


// get all files in a directory recurrently
void get_all_file_path_in_directory(std::vector<string> &out, const string &directory)
{
  DIR *dir;
  struct dirent *ent;
  struct stat st;
  if((dir = opendir(directory.c_str()))==NULL){
    cout<<"error open dir"<<std::endl;
  }
  while((ent=readdir(dir))!=NULL){
    const string file_name = ent->d_name;
    const string full_file_name = directory + '/' + file_name;

    if(file_name[0] == '.')
      continue;

    if(stat(full_file_name.c_str(), &st) == -1)
      continue;

    const bool is_directory = (st.st_mode & S_IFDIR) !=0;

    if(is_directory){
      get_all_file_path_in_directory(out, full_file_name);
      continue;
    }

    out.push_back(full_file_name);
  }
  closedir(dir);
}

string get_label_by_name(string path)
{
  int start=0, end= path.size()-4;
  bool flag=false;
  if(path[end] == '.'){
    for(start=end;start>=0;start--){
      if(path[start]=='_'){
	flag=true;
	break;
      }
    }
  }
  if(!flag){
    cout<<path<<" path name format error!\n";
    start=end;
  }
  string label;
  for(int i=start+1; i<end;i++){
    label.push_back(path[i]);
  }
  return label;
}
// merge the imdb data to two file: a data and a label				
void  merge_imdb_data(string source_folder_path, string merge_data_path, string merge_data_label_path)	
{
  std::vector<string> files_path;
  
  // read all files in one directory
  get_all_file_path_in_directory(files_path, source_folder_path);

  ofstream out_data(merge_data_path.c_str(), std::ofstream::out);
  ofstream out_label(merge_data_label_path.c_str(), std::ofstream::out);

  if(!out_data){
    std::cout<<"can't open "<<merge_data_path<<std::endl;
    exit(1);
  }

  if(!out_label){
    std::cout<<"can't open "<<merge_data_label_path<<std::endl;
    exit(1);
  }
    

  string label;

  float cur_percent=1;
  int file_num = files_path.size();
  for(int i=0; i<file_num;i++){

    if(i*100.0/file_num>cur_percent){
      cur_percent = i*100.0/file_num;
      cout<<"percent:"<<cur_percent<<endl;
      cur_percent+=1;
    }
    const string file_path = files_path[i]; //get one file path
    // read data from file
    std::ifstream is(file_path.c_str());
    
    if(is){
      //get length of file:
      is.seekg(0, is.end);
      int length = is.tellg();
      is.seekg(0, is.beg);

      char* buffer = new char [length+1];

      is.read(buffer, length);
      buffer[length]='\n';

      if(!is)
	std::cout<<"error:only "<<is.gcount()<<" could be read";

      out_data.write(buffer,length+1);

      label = get_label_by_name(file_path);
      
      out_label<<label<<endl;

      is.close();

      delete[] buffer;
     }
  }
  out_data.close();
  out_label.close();
}


int main()
{

   //imdb test data merge
   string test_source_folder_path = "./data/aclImdb/test";
   string test_merge_data_path = "./data/aclImdb/test-merge/test.data";
   string test_merge_data_label_path="./data/aclImdb/test-merge/test.label";
   merge_imdb_data(test_source_folder_path, test_merge_data_path, test_merge_data_label_path);

  
  //imdb train data merge
   string train_source_folder_path = "./data/aclImdb/train";
   string train_merge_data_path = "./data/aclImdb/train-merge/train.data";
   string train_merge_data_label_path="./data/aclImdb/train-merge/train.label";
   merge_imdb_data(train_source_folder_path, train_merge_data_path, train_merge_data_label_path);
}
