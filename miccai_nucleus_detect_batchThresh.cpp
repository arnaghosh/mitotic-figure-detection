#include <bits/stdc++.h>
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
			string sys_s2 = "python reg2hedTest.py /media/arna/HP_32/Arna_Dataset/segmentation_training/"+folderName[i]+"/training-set/"+imName+" /media/arna/HP_32/Arna_Dataset/segmentation_training/"+folderName[i]+"/maskedRes/"+imName;
			system(sys_s2.c_str());
		}
	}

	return 0;
}