#include <bits/stdc++.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
using namespace std;

cv::Mat rotateImage(const cv::Mat source, double angle,int border=20)
{
    cv::Mat bordered_source;
    int top,bottom,left,right;
    top=bottom=left=right=border;
    cv::copyMakeBorder( source, bordered_source, top, bottom, left, right, cv::BORDER_CONSTANT,cv::Scalar() );
    cv::Point2f src_center(bordered_source.cols/2.0F, bordered_source.rows/2.0F);
    cv::Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
    cv::Mat dst;
    cv::warpAffine(bordered_source, dst, rot_mat, bordered_source.size());  
    return dst;
}

int main(){
	string folderName[11] = {"A03","A04","A05","A07","A10","A11","A12","A14","A15","A17","A18"};
	for(int f=0;f<11;f++){
		cout<<"Set "<<f<<" running..."<<endl;
		int count=1;
		int mit_count=0;
		string path="/home/siplab/mitotic-figure-detection/Training_Data/Gen_Dataset/"+folderName[f]+"/mitosis/";
		string trainPath = "/home/siplab/mitotic-figure-detection/Fold_Train_few/"+folderName[f]+"/mitosis/";
		string cmd = "ls "+path+" > tempTxt.txt";
		system(cmd.c_str());
		ifstream file1("tempTxt.txt");
		string imName;
		while(file1.good()){
			getline(file1,imName);
			if(imName.empty())continue;
			mit_count++;
			string imNameTot = path+""+imName;
			cv::Mat im = cv::imread(imNameTot,1);
			// cv::imshow("im",im);
			// cv::waitKey(0);
			for(int i=0;i<10;i++){
				cv::Mat im2 = rotateImage(im,36*i,0);
				// cv::imshow("im2",im2);
				// cv::waitKey(0);
				cv::imwrite(trainPath+to_string(count)+".jpg",im2);
				count++;
				cv::flip(im2,im2,1);
				// cv::imshow("im2",im2);
				// cv::waitKey(0);
				cv::imwrite(trainPath+to_string(count)+".jpg",im2);
				count++;
			}
		}
		cout<<"before= "<<mit_count<<" ,now= "<<(mit_count*20)<<endl;
		cout<<"mitosis done!! Calculating required Rotations for non-mitotic.."<<endl;
		string path2="/home/siplab/mitotic-figure-detection/Training_Data/Gen_Dataset/"+folderName[f]+"/not_mitosis/";
		string trainPath2 = "/home/siplab/mitotic-figure-detection/Fold_Train_few/"+folderName[f]+"/not_mitosis/";
		cmd = "ls "+path2+" > tempTxt.txt";
		system(cmd.c_str());
		ifstream file2("tempTxt.txt");
		int nmit_count=0;
		while(file2.good()){
			getline(file2,imName);
			if(imName.empty())continue;
			nmit_count++;
		}
		cout<<nmit_count<<endl;
		int reqRot = (10*mit_count)/nmit_count;
		double rotAngle = (360.0/reqRot);
		cout<<reqRot<<" "<<rotAngle<<endl;
		ifstream file3("tempTxt.txt");
		count = 1;
		while(file3.good()){
			getline(file3,imName);
			if(imName.empty())continue;
			string imNameTot = path2+""+imName;
			cv::Mat im = cv::imread(imNameTot,1);
			for(double i=0;i<360.0;i=i+rotAngle){
				cv::Mat im2 = rotateImage(im,i,0);
				cv::imwrite(trainPath2+to_string(count)+".jpg",im2);
				count++;
				cv::flip(im2,im2,1);
				cv::imwrite(trainPath2+to_string(count)+".jpg",im2);
				count++;
			}
		}
		cout<<"Non-mitosis done!!"<<endl;
	}
	return 0;
}