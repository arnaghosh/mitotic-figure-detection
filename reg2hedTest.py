import matplotlib.pyplot as plt
from skimage.color import rgb2hed
import cv2
import numpy as np
import sys

print("python: ",str(sys.argv[1]))
img = cv2.imread(str(sys.argv[1]),1)
#img = cv2.imread("Training Data/A03/frames/x40/A03_00Aa.tiff",1)
#img = cv2.pyrDown(img)
ihc_hed = rgb2hed(img)

min,max,minLoc,maxLoc = cv2.minMaxLoc(ihc_hed[:,:,2])
print min,max

ihc_hed = np.array(ihc_hed,dtype = 'float32');
ret,thresh = cv2.threshold(ihc_hed[:,:,2],min+(max-min)*0.5,255,cv2.THRESH_BINARY);
#thresh = ihc_hed[:,:,2]
#ret = 0
kernelSize = 5
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
#thresh = cv2.erode(thresh,kernel,iterations=2)
#thresh = cv2.dilate(thresh,kernel,iterations=2)
#thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations=1)
thresh = cv2.convertScaleAbs(thresh);thresh = cv2.medianBlur(thresh,9)
#print ret

#cv2.imshow("1",thresh)
cv2.imwrite(sys.argv[2],thresh)
#cv2.waitKey(0)

fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
ax0, ax1, ax2, ax3 = axes.ravel()

ax0.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax0.set_title("Original image")

ax1.imshow(ihc_hed[:, :, 0], cmap=plt.cm.gray)
ax1.set_title("Hematoxylin")

ax2.imshow(ihc_hed[:, :, 1], cmap=plt.cm.gray)
ax2.set_title("Eosin")

#ax3.imshow(ihc_hed[:,:,2], cmap=plt.cm.gray)
ax3.imshow(thresh, cmap=plt.cm.gray)
ax3.set_title("Residual")

for ax in axes.ravel():
    ax.axis('off')

fig.subplots_adjust(hspace=0.3)
#plt.show()




