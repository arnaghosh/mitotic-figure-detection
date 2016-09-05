#include <bits/stdc++.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
using namespace std;

int main(){
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
			cv::Mat im = cv::imread(imNameTot,1);
			cv::imshow("im",im);
			cv::waitKey(0);
			cv::balanceWhite(im,im,cv::CV_WHITE_BALANCE_SIMPLE);
			cv::imshow("im",im);
			cv::waitKey(0);
		}
	}

	return 0;
}