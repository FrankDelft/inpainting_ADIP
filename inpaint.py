import numpy as np
from  scipy.misc import derivative
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
def in_paint_alg(img, target_region,source_region,patch_size):
    im_x=img.shape[0]
    im_y=img.shape[1]
    C=np.zeros((im_x,im_y))
    D=np.zeros((im_x,im_y))
    isophote=isophote(img,0.25)

    


#function  found in matlab documentation
def isophote(L, alpha):
    L = L.astype(float) / 255.0
    theta = np.zeros_like(L)
    Lx, Ly = np.gradient(L)
    I = np.sqrt(Lx**2 + Ly**2)
    I = I / np.max(np.max(I))
    print(np.max(np.max(I)), "\tmax all\t",np.max(I))
    T = (I >= alpha)
    T_bin = T.astype(int)
    theta[T] = np.arctan2(Ly[T], Lx[T])
    theta_shape = theta.shape
    theta_modified = np.zeros(theta_shape)
    # for i in range(theta_shape[0]):
    #     for j in range(theta_shape[1]):
            # x = theta[i, j]
            # if x > np.pi/2:
            #     theta_modified[i, j] = np.pi/2 - x
            # elif x <-np.pi/2:
            #     theta_modified[i, j] = -1*(x + np.pi/2)
            # elif x==0:
            #     theta_modified[i, j] = x
            # else:
            #     theta_modified[i, j] = np.pi/2-x
    # theta = theta_modified
    theta = theta * 180 / np.pi
    I[I < alpha] = 0
    return I, theta




# # Define your continuous function
# def f(x):
#     return x

# # Define the derivative function
# def df(x):
#     return derivative(f, x, dx=1e-8)

