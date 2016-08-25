
#include <cv.h>
#include <highgui.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <strings.h>
#include <map>
#include <utility>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
using namespace std;

#define pii pair<int,int>

int main(int argc,char** argv){
	if(argc!=4){
		cout<<"\nArgument list: \n";
		cout<<"1. Training or Testing Data -> 0 for training and 1 for testing.\n";
		cout<<"2. Folder to extract image from. eg - A03 or A04.\n";
		cout<<"3. Mitosis or mon-mitosis -> 0 for mitosis, 1 for non-mitosis. \n\n";
		return 0;
	}
	string TrTeName[2] = {"Training Data","Testing Data"};
	string mit[2] = {"mitosis","not_mitosis"};
	string imName = TrTeName[atoi(argv[1])]+"/"+argv[2]+"/frames/x40/"+argv[2]+"_00Aa.tiff";
	cv::Mat img = cv::imread(imName,1);
	cv::Mat bin = cv::imread("1.jpg",0);
	string filename = "Training Data/A03/mitosis/A03_00Aa_mitosis.csv";
	cv::imshow("img",img);
	cv::imshow("bin",bin);
	cv::waitKey(0);

	ifstream file(filename);
	string val;
	vector<pii> P;
	while(file.good()){
		pii p;
		for(int i=0;i<3;i++){
			getline(file,val,',');
			if(i==0)p.first = stoi(val);
			if(i==1)p.second = stoi(val);
		}
		P.push_back(p);
	}
	//for(unsigned int i=0;i<P.size();i++)cout<<P[i].first<<" "<<P[i].second<<endl;
	int patchSize = 101;
	cv::Mat labels(img.rows,img.cols, CV_32SC1);
	int connectedComp = cv::connectedComponents(bin,labels,8);
	vector<cv::Vec3b> colors(connectedComp);
	colors[0]=cv::Vec3b(0,0,0);
	for(int i = 1; i< connectedComp; i++){
        colors[i] = cv::Vec3b( (rand()&255), (rand()&255), (rand()&255) );
	}
	int imagesGenerated = 0;
	string folder = "Training Data/Gen_Dataset/mitosis/";
	cv::Mat dst(img.rows,img.cols,CV_8UC3);
	for(int i=0;i<img.rows;i=i+5){
		for(int j=0;j<img.cols;j=j+5){
			int label = labels.at<int>(i,j);
			dst.at<cv::Vec3b>(i,j) = colors[label];
			if(label==0)continue;
			int x = i-patchSize/2;
			int y = j-patchSize/2;
			if(x<0)x=0;
			if(y<0)y=0;
			if(x+patchSize>img.rows)x=img.rows-patchSize-1;
			if(y+patchSize>img.cols)y=img.cols-patchSize-1;
			int grTrKnwon=0;
			for(unsigned int k=0;k<P.size();k++){
				if(P[k].first>=y && P[k].first<=y+patchSize && P[k].second>=x && P[k].second<=x+patchSize){grTrKnwon=1;break;}
			}
			if(!grTrKnwon)continue;
			cv::Mat temp = img(cv::Rect(y,x,patchSize,patchSize));
			string imName = folder+to_string(imagesGenerated)+".jpg";
			cout<<imName<<endl;
			cv::imwrite(imName,temp);
			imagesGenerated++;
		}
	}
	cv::imshow("dst",dst);
	cv::waitKey(0);
	return 0;
}