#include <bits/stdc++.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
using namespace std;

void whiteBalanced(cv::Mat& im, double discardRatio){
	long long int hist[3][256];
	for(int i=0;i<3;i++){
		for(int j=0;j<256;j++)hist[i][j]=0;
	}
	for(int i=0;i<im.rows;i++){
		for(int j=0;j<im.cols;j++){
			hist[0][(int)im.at<cv::Vec3b>(i,j)[0]]++;
			hist[1][(int)im.at<cv::Vec3b>(i,j)[1]]++;
			hist[2][(int)im.at<cv::Vec3b>(i,j)[2]]++;
		}
	}
	long long int total = im.rows*im.cols;
	int Min[3],Max[3];
	for(int i=0;i<3;i++){
		for(int j=1;j<256;j++){
			hist[i][j]+=hist[i][j-1];
		}
		Min[i]=0;Max[i]=255;
		while(hist[i][Min[i]]<discardRatio*total)Min[i]++;
		while(hist[i][Max[i]]>(1-discardRatio)*total)Max[i]--;
		if(Max[i]<255-1)Max[i]++;
	}
	for(int i=0;i<im.rows;i++){
		for(int j=0;j<im.cols;j++){
			for(int k=0;k<3;k++){
				int val = (int)im.at<cv::Vec3b>(i,j)[k];
				if(val<Min[k])val = Min[k];
				if(val>Max[k])val = Max[k];
				im.at<cv::Vec3b>(i,j)[k] = (int)((val-Min[k])*255.0/(Max[k]-Min[k]));
			}
		}
	}
}

int main(int argc, char** argv){
	if(argc<2){
		cout<<"\nArgument List:\n1. Threshold fraction of pixels to be ignored for white balancing\n";
	}
	string folderName[4] = {"gbm","hnsc","lgg","lung"};
	string imFolderName = "training-set";
	string path = "/media/arna/HP_32/Arna_Dataset/segmentation_training/";
	for(int i=0;i<4;i++){
		string sys_s = "ls "+path+folderName[i]+"/training-set/ | grep -v _ > ls_res.txt";
		system(sys_s.c_str());
		ifstream ls_res("ls_res.txt");
		string imName;
		while(ls_res.good()){
			getline(ls_res,imName);
			if(imName.empty())continue;
			string imNameTot = "/media/arna/HP_32/Arna_Dataset/segmentation_training/"+folderName[i]+"/training-set/"+imName;
			string imNameTot2 = "/media/arna/HP_32/Arna_Dataset/segmentation_training/"+folderName[i]+"/whiteBalanced-set/"+imName;
			cout<<imNameTot2<<endl;
			cv::Mat im = cv::imread(imNameTot,1);
			/*cv::imshow("im",im);
			cv::waitKey(0);*/
			whiteBalanced(im,atof(argv[1]));
			/*cv::imshow("im",im);
			cv::waitKey(0);*/
			cv::imwrite(imNameTot2,im);
		}
	}

	return 0;
}