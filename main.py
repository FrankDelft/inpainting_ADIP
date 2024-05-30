import diplib as dp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from inpaint import *
import inpaint
from util import *

# Load the image
name='resources/image1.jpg'
name_mask='resources/mask11.jpg'


img = cv2.imread(name)
img_mask=cv2.imread(name_mask,cv2.IMREAD_GRAYSCALE)
img_mask = cv2.bitwise_not(img_mask)
img_mask = cv2.threshold(img_mask, 127, 255, cv2.THRESH_BINARY)[1]


# source_region_mask = cv2.bitwise_not(img_mask)
source_region_mask = (img_mask)
source_indices_mask=np.array(np.where(source_region_mask==255)).T


inpaint_obj = inpaint.Inpainting(img, source_indices_mask, patch_size=1)
inpaint_obj.in_paint_alg()