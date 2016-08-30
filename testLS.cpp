#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>
/*#include <cv.h>
#include <highgui.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"*/
using namespace std;

int main(){
	string s ="Training_Data/A03/frames/x40";
	string s2 ="mitosis";
	//s = "ls "+s+" | grep ^A > ls_res.txt";
	s = "ls "+s+" > ls_res.txt";
	system(s.c_str());
	ifstream file("ls_res.txt");
	string line;
	while(file.good()){
		getline(file,line);
		if(line.empty())continue;
		cout<<line<<" ";
		string line2 = line; line2.replace(line2.end()-5,line2.end(),"_"+s2+".csv");
		cout<<line2<<endl;
		/*cv::Mat im = cv::imread(line,1);
		cv::imshow("im",im);
		cv::waitKey(0);*/
	}
	return 0;
}