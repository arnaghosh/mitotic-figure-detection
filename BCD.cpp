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
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
using namespace std;

#define pdd pair<double,double>

pdd rgb2maxwell(int r, int g, int b){
	pdd a;
	if(r==0)r=1;
	if(g==0)g=1;
	if(b==0)b=1;
	double x1 = log2(255.0/r);
	double x2 = log2(255.0/g);
	double x3 = log2(255.0/b);
	double mag = sqrt(x1*x1 + x2*x2 + x3*x3);
	if(mag<=0.0001){x1=0;x2=0;x3=0;}
	else {x1/=mag; x2/=mag; x3/=mag;}
	//cout<<r<<" "<<g<<" "<<b<<" "<<x1<<" "<<x2<<" "<<x3<<" "<<mag<<endl;
	a.first = (x1-x2)/sqrt(2);
	a.second = -1.0*(x1+x2)/sqrt(6) + x3*sqrt(2/3);
	//cout<<a.first<<" "<<a.second<<endl;
	return a;
}

pdd round(pdd a){
	long int x = (int)(a.first*100);
	long int y = (int)(a.second*100);
	pdd b;
	b.first = (double)1.0*x/100.0;
	b.second = (double)1.0*y/100.0;
	return b;
}

void showMaxwellTriangle(map<pdd,long long int> M){
	cv::Mat im(433,500,CV_8UC1,cv::Scalar(0));
	cv::line(im,cv::Point(250,0),cv::Point(0,433),255,1);
	cv::line(im,cv::Point(250,0),cv::Point(500,433),255,1);
	cv::line(im,cv::Point(0,432),cv::Point(500,432),255,1);
	double a11 = 250*sqrt(2);
	double a12 = 0.0;
	double c1 = 250.0;
	double a21 = 0.0;
	double a22 = -432*sqrt(2.0/3.0);
	double c2 = 288 ; 
	long long int maxCount = 0;
	for(map<pdd,long long int>::iterator it=M.begin();it!=M.end();++it){
		if(it->second > maxCount)maxCount = it->second;
	}
	for(map<pdd,long long int>::iterator it=M.begin();it!=M.end();++it){
		it->second = (int)(it->second*255.0/maxCount);
		if(it->second>255)it->second = 255;
	}
	for(map<pdd,long long int>::iterator it=M.begin();it!=M.end();++it){
		double a=it->first.first;
		double b=it->first.second;
		double x = a11*a + a12*b + c1;
		double y = a21*a + a22*b + c2;
		if((int)im.at<uchar>((int)y,(int)x)<it->second)im.at<uchar>((int)y,(int)x) = (int)it->second;
		//im.at<uchar>((int)y,(int)x) = 255;
		cout<<x<<" "<<y<<" "<<it->second<<endl;
	}
	cv::imshow("maxwell",im);
	cv::waitKey(0);
}

int main(int argc, char** argv){
	cv::Mat im = cv::imread(argv[1],1);
	cv::imshow("image",im);
	cout<<"image size: "<<im.rows<<" X "<<im.cols<<endl;
	cv::waitKey(0);
	map<pdd,long long int> maxwellMap;
	for(int i=0;i<im.rows;i++){
		for(int j=0;j<im.cols;j++){
			pdd a = rgb2maxwell((int)im.at<cv::Vec3b>(i,j)[2],(int)im.at<cv::Vec3b>(i,j)[1],(int)im.at<cv::Vec3b>(i,j)[0]);
			a = round(a);
			if(maxwellMap.count(a)==0)maxwellMap.insert(make_pair(a,1));
			else maxwellMap[a]+=1;
		}
	}
	/*for (map<pdd,long long int>::iterator it=maxwellMap.begin(); it!=maxwellMap.end(); ++it)
    	cout <<"("<< it->first.first <<","<<it->first.second<<")" << " => " << it->second << endl;*/
	showMaxwellTriangle(maxwellMap);
	return 0;
}