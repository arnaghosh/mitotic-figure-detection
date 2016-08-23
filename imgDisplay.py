import cv2
import numpy

img = cv2.imread("Training Data//A03/mitosis/A03_00Aa_mitosis.jpg",1)
#cv2.circle(img,(1094,1223),10,(255,0,0),5)
print(img.shape)
img2 = img[0:101,0:101]
print(img2.shape)
img = cv2.pyrDown(img)

cv2.imshow("img",img)
cv2.imshow("img2",img2)
cv2.waitKey(0);
cv2.imwrite("trainImg1.jpg",img2)