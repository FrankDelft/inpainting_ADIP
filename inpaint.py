import numpy as np
from  scipy.misc import derivative
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
def in_paint_alg(img, function,target_region,source_region,patch_size):
    im_x=img.shape[0]
    im_y=img.shape[1]
    C=np.zeros((im_x,im_y))
    




# # Define your continuous function
# def f(x):
#     return x

# # Define the derivative function
# def df(x):
#     return derivative(f, x, dx=1e-8)

