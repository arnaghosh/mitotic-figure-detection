
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

int main(){
	cv::Mat img = cv::imread("Training Data/A03/frames/x40/A03_00Aa.tiff",1);
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