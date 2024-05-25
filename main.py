import diplib as dp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from inpaint import *
import inpaint
from util import *

# Load the image
name='girl.png'
name_mask='girl2.png'
img = cv2.imread(name)
img_mask=cv2.imread(name_mask,cv2.IMREAD_GRAYSCALE)

img_col = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.imread(name, cv2.IMREAD_GRAYSCALE)

#rectangle def
x_width = img_gray.shape[1]
y_height = img_gray.shape[0]
x1,y1,x2,y2=(120,280,230,400)
#circle def
x1_c,y1_c,r=(150,290,50)


######################################################
contour_indices_rect,source_indices_rect,target_indices_rect= rectangle_target(x_width,y_height,x1,y1,x2,y2)
contour_indices_circ,source_indices_circ,target_indices_circ= circle_target(x_width,y_height,x1_c,y1_c,r)

source_region_mask = cv2.bitwise_not(img_mask)
source_indices_mask=np.array(np.where(source_region_mask==255)).T
contour_mask=get_contour(source_region_mask)

P=in_paint_alg(img,source_indices_mask,patch_size=4)
