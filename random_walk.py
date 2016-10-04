import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import random_walker
import skimage
import cv2
import sys

im = cv2.imread(str(sys.argv[1]),1)
binIm = cv2.imread(str(sys.argv[2]),0)
cv2.pyrDown(im,im);
cv2.pyrDown(binIm,binIm);
comp = cv2.connectedComponentsWithStats(binIm,8,cv2.CV_16U)
#labels = random_walker(im,markers,)
print comp[0]
labelIm = cv2.convertScaleAbs(comp[1]);
markers = np.zeros(im.shape, dtype = np.uint)
for i in range(1,len(comp[3])):
	cv2.circle(markers,)
labels = random_walker(im, markers, beta=10, mode='bf')
cv2.imshow("im",im);
cv2.imshow("labels",labelIm);
cv2.waitKey(0);


'''import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import random_walker
#from skimage.data import binary_blobs
import skimage
import sys
import cv2
# Generate noisy synthetic data
#data = skimage.img_as_float(binary_blobs(length=128, seed=1))
#data += 0.35 * np.random.randn(*data.shape)
data = cv2.imread(str(sys.argv[1]),1);
binIm = cv2.imread(str(sys.argv[2]),0);
markers = np.zeros(data.shape, dtype=np.uint)
markers[binIm < 100] = 1
markers[binIm > 100] = 2

# Run random walker algorithm
labels = random_walker(data, markers, beta=10, mode='bf')

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
                                    sharex=True, sharey=True)
ax1.imshow(binIm, cmap='gray', interpolation='nearest')
ax1.axis('off')
ax1.set_adjustable('box-forced')
ax1.set_title('Noisy data')
ax2.imshow(markers, cmap='hot', interpolation='nearest')
ax2.axis('off')
ax2.set_adjustable('box-forced')
ax2.set_title('Markers')
ax3.imshow(labels, cmap='gray', interpolation='nearest')
ax3.axis('off')
ax3.set_adjustable('box-forced')
ax3.set_title('Segmentation')

fig.tight_layout()
plt.show()
'''