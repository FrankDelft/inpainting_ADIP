import diplib as dp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from inpaint import *
import inpaint

img = cv2.imread('test.png')
img_col = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)

x_width = img_gray.shape[1]
y_height = img_gray.shape[0]
x1,y1,x2,y2=(120,280,230,400)
print("x_width",x_width,"y_height",y_height)

plt.imshow(img_col)
plt.axis('off')
plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='r', facecolor='none'))
plt.show()


#######################################################
#define contour and regions
image = np.zeros((x_width, y_height), np.uint8) 
thickness = 1
contour = cv2.rectangle(image.copy(),[y1,x1],[y2,x2], 255, thickness) 
# Creating rectangle
thickness = -1
target_region = cv2.rectangle(image.copy(),[y1,x1],[y2,x2], 255, thickness) 
thickness = -1
source_region =np.array(cv2.bitwise_not(target_region))

#get x and y pixel co-ordinates
temp=np.where(np.array(np.array(contour)==255,dtype=int)==1)
contour_indices = np.array(list(zip(temp[0],temp[1])))
temp=np.where(np.array(np.array(source_region)==255,dtype=int)==1)
#source indices
source_indices = np.array(list(zip(temp[0],temp[1])))
temp=np.where(np.array(np.array(target_region)==255,dtype=int)==1)
# target_indices = np.array(list(zip(temp[0],temp[1])))


P=in_paint_alg(img,contour_indices,source_indices)
