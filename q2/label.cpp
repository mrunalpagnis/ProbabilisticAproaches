#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <sstream>
#include <set>
#include <algorithm>
#include <numeric>
#include <string.h>

using namespace std;

typedef unordered_map<string, double> Row;
typedef unordered_map<string, unordered_map<string, double> > Table;

#define MAX_SEN_LEN 100
#define TOTAL "total"
#define NAIVE "Part 2, Naive graphical model"
#define BAYES "Part 3, Bayes net"
#define SAMPLE "Part 4, Sampling"

typedef unordered_map<string, double> stateMap;
Row s1;					//probabilities of states occuring in first place of sentence
Row Si;					//probabilities of states in total number of words
Table sM;				//probabilities Si+1 given Si
Table stateWTable;		//probabilities of words given a state
unordered_map<int,unordered_map<string,double>> posterior;
unordered_map<int,string> maximum;

long total_word_count=0;
long sentence_count = 0;

/***
initializing the table s1 with all counts set to zero
list of 12 part of speech is stored in file tags.txt 
***/
void initializeS1(ifstream &ifs)
{
	s1.clear();
	while(ifs.good()){
		string line, label;
		getline(ifs,line);
		if(!ifs.good() || line == "")
		break;
		istringstream iss(line);
		iss>>label;
		
		if(label.size() < 4)
	for(int i=label.size(); i<4; i++)
	  label = label + " ";
		
		s1.insert(pair<string,double>(label,0.0));
		Si.insert(pair<string, double>(label,0.0));
	}
	
	//no need of end state
	/*s1.insert(pair<string,double>("E",0.0000000001));
	Si.insert(pair<string,double>("E",0.0000000001));*/
}

bool read_sentence(ifstream &ifs, vector<string> &sentence, vector<string> &labels) {
  labels.clear();
  sentence.clear();
	bool w1 = true;
  while(ifs.good()) {
      string line, word, label;
      getline(ifs, line);
      if(!ifs.good() || line == "")
	break;
      istringstream iss(line);

      iss >> word >> label;
	
      if(label.size() < 4)
	for(int i=label.size(); i<4; i++)
	  label = label + " ";
	  
	  /*plug in the counts for each part of speech occurring at starting of the sentence*/
		if(w1) //if first word
		{
			s1[label]++; //counting #(s1)
			w1 = false;
		}
		else  //if word in other position
		{
			sM[label][labels.back()]++; //counting #(Si+1,Si)
		}
		Si.at(label) = Si.find(label)->second++; //#(Si)
		stateWTable[label][word]++;	//counting #(W,S)
      sentence.push_back (word);
      labels.push_back(label);
  }

  return sentence.size() > 0;
}

/*reading the test data 
this function is different from the read_sentence function because the read_sentence function calculates the counts as well
However this function it is not required to do the counts on test data*/
bool read_test_sentence(ifstream &ifs, vector<string> &sentence, vector<string> &labels) {
  labels.clear();
  sentence.clear();
	bool w1 = true;
  while(ifs.good()) {
      string line, word, label;
      getline(ifs, line);
      if(!ifs.good() || line == "")
	break;
      istringstream iss(line);

      iss >> word >> label;
		
      if(label.size() < 4)
	for(int i=label.size(); i<4; i++)
	  label = label + " ";
	  sentence.push_back (word);
      labels.push_back(label);
  }

  return sentence.size() > 0;
}
template<class T>
ostream &operator<<(ostream &os, const vector<T> &vec){
  for(int i=0; i<(int)vec.size(); i++)
    os << vec[i] << " ";

  return os;
}

/* funtion for Max marginal inference */
void max_marginal_inference(vector<string> sentence)
{
	int len = sentence.size();
	unordered_map<int,unordered_map<string,double>> fwd;
	unordered_map<int,unordered_map<string,double>> bkw;
	
	/*forward algorithm*/
	Row f_prev;
	for(int i=0;i<sentence.size();i++)
	{
		Row f_curr;
		for(Row::const_iterator iter=s1.begin(); iter != s1.end(); ++iter )
		{
			double prev_f_sum = 0;	
			if(i==0)
			{
				prev_f_sum = iter->second;
			}
			else
			{
				double sum = 0;
				for(Row::const_iterator it=s1.begin(); it != s1.end(); ++it)
				{
					if(sM[iter->first][it->first]==0)
						sM[iter->first][it->first]=0.00000001; //values that do not exist
					sum += (f_prev[it->first]*sM[iter->first][it->first]);
				}
				prev_f_sum = sum;
			}
			double y = stateWTable[iter->first][sentence[i]];
			f_curr[iter->first] = y*prev_f_sum;
		}
		f_prev.clear();
		string str = sentence[i];
		for(Row::const_iterator iter=f_curr.begin(); iter != f_curr.end(); ++iter)
		{ 
			fwd[i][iter->first] = iter->second;
			f_prev[iter->first] = iter->second; 
		}
	}
	double p_fwd=0;
	for(Row::const_iterator iter=f_prev.begin(); iter != f_prev.end(); ++iter)
	{
		double y =  0.000000000001; //values that do not exist
		p_fwd += (iter->second * y);
	}
	
	/*backward algorithm*/
    Row b_prev;    
	for(Row::const_iterator iter=s1.begin(); iter != s1.end(); ++iter )
		b_prev[iter->first] = 1.0;
	for(int i=len-1;i>=0;i--)
	{
		Row b_curr;
		for(Row::const_iterator iter=s1.begin(); iter != s1.end(); ++iter )
		{			
			
				double sum = 0;
				for(Row::const_iterator it=s1.begin(); it != s1.end(); ++it)
				{
					if(sM[it->first][iter->first]==0.0)
						sM[it->first][iter->first] = 0.0000000001;  //minute value that does not affect
					double y = stateWTable[it->first][sentence[i]];
					if(y == 0)
						y = (double)1/(double)12;
					sum+= (sM[it->first][iter->first]*y*b_prev[it->first]);
				}
				
				b_curr[iter->first] = sum;
			
		}
		string str = sentence[i];
		for(Row::const_iterator iter=b_curr.begin(); iter != b_curr.end(); ++iter)
		{
			bkw[i][iter->first] = iter->second; 
			b_prev[iter->first] = iter->second;
		}
	}
	
	/*calculating the posterior*/
	for(int i=0,j=len-1;i<len && j>=0 ;i++, j--)
	{
		double max = 0.0;
		string label="";
		for(Row::const_iterator it=s1.begin(); it != s1.end(); ++it)
		{
			posterior[i][it->first] = fwd[i][it->first]*bkw[j][it->first];
			if(max<posterior[i][it->first])  //computing the max posterior for word at each position
			{
				max = posterior[i][it->first];		
				label = it->first;
			}
		}
		maximum.insert(pair<int,string>(i,label));
	}
	
}

/*calculates probabilities from the counts*/
void calculate_probalities()
{
	for(Row::iterator iter=s1.begin();iter!=s1.begin();++iter)
	{
		double x = iter->second;
		s1[iter->first] = x/sentence_count;
	}
	for(Table::iterator iter = sM.begin(); iter!=sM.end(); ++iter)
	{
		for(Row::iterator it=iter->second.begin(); it != iter->second.end(); ++it)
		{
			
			sM[iter->first][it->first] = (it->second) / Si[it->first];
		}
	}
	for(Table::iterator iter = stateWTable.begin(); iter!=stateWTable.end(); ++iter)
	{
		for(Row::iterator it=iter->second.begin(); it != iter->second.end(); ++it)
		{
			stateWTable[iter->first][it->first] = it->second / Si[iter->first];
		}
	}
}
void print_sample(vector<string> &sentence, vector<string> &labels, int &cw, int &cs)
{
	double n= 5.0;
	double ptr;
	int correct_words[5];
	for(int i=0;i<5;i++)
	{
		cout<<"SAMPLE "<<i<<": ";
		for(int j=0;j<sentence.size();j++)
		{
			string label1 = "",label2="";
			if(n<3)
			{
			double max1=0;double max2=0;
			label2 = "";
			for(Row::iterator it = s1.begin();it!=s1.end();++it)
			{
				if(max1<posterior[j][it->first])
				{
					label2 = label1;
					max2 = max1;
					max1 = posterior[j][it->first];
					label1 = it->first;					
				}
			}
			n--;
			}
			label2 = maximum[j];
			cout<<label2<<" ";	
			if(label2 == labels[j])
				correct_words[i]++;
		}
		if(correct_words[i] == sentence.size()-1)
		{
			cs++;
			cw = sentence.size()-1;
		}
		cout<<endl;
	}
	
}

int main(int argc, char *argv[]){

  if (argc < 3) {
    printf( "\nusage: ./%s train-file test-file\nwhere the files are of the format\n", argv[0]);
    printf("The DET\nmonkey NOUN\nis VERB\nhappy ADJ\n'' .\n");
    exit(-1);
  }
  
  try {
		/*    To get P(s1) 
	set initial counts to zero
	tags.txt has the shortforms of 12 parts of speech*/
	ifstream ifs3("tags.txt");
	initializeS1(ifs3);
    
	/* 
     * Currently the below simply reads through the training file
     * and outputs the number of words and sentences.
     * You need to make it learn conditional probabilities.
     */
    ifstream ifs(argv[1]);

    vector<string> sentence, labels;
	cout<< "calculating ... "<<endl;
    while(read_sentence(ifs, sentence, labels)){
	sentence_count++;

	for(int i=0; i<sentence.size(); i++){
	    total_word_count++;
	}
    }

    printf("In the training file, the number of words is %lu and the number of sentences is %lu\n", total_word_count, sentence_count);
	/*
     * Now run on test data.
     * Currently again only counts the number of words and sentences.
     * You need to make the algorithms run on the test data.
     */
    ifstream ifs2(argv[2]);
    Row correct, sentence_correct;
	
	/*calculating the probabilities from counts*/
	calculate_probalities();
	
	while(read_test_sentence(ifs2, sentence, labels)){
		cout << "--------------" << endl;
		cout << "Considering sentence: " << sentence << endl;
		cout << "Ground truth: " << labels << endl;
		cout << "Naive: ";
		int cw = 0;
		for(int i=0; i<sentence.size(); i++) {
			double max=0;
			string label_i="";
			for(Row::const_iterator iter=Si.begin(); iter != Si.end(); ++iter)
			{
				double p = stateWTable[iter->first][sentence[i]];
				if(p == 0)
				p = 1/12;
				p = (p*(iter->second));
				if(p>max)
				{
					max = p;
					label_i = iter->first;
				}
			}
			cout<<label_i<<" ";
			if(label_i == labels[i])
			{	
				cw++;
				correct[NAIVE]++;
			}
			correct[TOTAL]++;
		}
		cout<<endl;
		if(cw == sentence.size()-1)
			sentence_correct[NAIVE]++;
		sentence_correct[TOTAL]++;
		
		/* Max marginal method*/
		posterior.clear();
		maximum.clear();
		max_marginal_inference(sentence);
		cout<<"BAYES: ";
		cw = 0;
		for(int j=0;j<sentence.size();j++)
		{
			cout<< maximum[j]<<" ";
			if(maximum[j] == labels[j]) 
			{
				cw++;
				correct[BAYES]++;
			}
		}
		cout<<endl;
		if(cw == sentence.size()-1)
			sentence_correct[BAYES]++;
		cout<<endl;
		int cs=0;
		cw = 0;
		print_sample(sentence,labels,cw,cs);
		correct[SAMPLE] += cw;
		sentence_correct[SAMPLE] += cs;
	}
    
    cout << "--------------" << endl;
    cout << "PERFORMANCE SUMMARY " << endl;
    cout << "Total words: " << correct[TOTAL] << "  sentences: " << sentence_correct[TOTAL] << endl;
    for(Row::const_iterator iter=correct.begin(); iter != correct.end(); ++iter){
	if(iter->first == TOTAL)
	  continue;
	cout << iter->first << ": " << endl;
	if(iter->first == SAMPLE)
	  cout << "(Here, sentences correct is fraction for which at least one sample is completely correct)" << endl;
	cout << "      Words correct: " << correct[iter->first] / double(correct[TOTAL]) << endl;
	cout << "  Sentences correct: " << sentence_correct[iter->first] / double(sentence_correct[TOTAL]) << endl;
    }
  }
  catch(const string &str) {
      cerr << "Error: " << str << endl;
  }
  
}
  
