import numpy as np
from  scipy.misc import derivative
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
def in_paint_alg(img, contour, source_region,normal,patch_size=9):
    im_x=img.shape[1]
    im_y=img.shape[0]
    C=source_region.copy()
    C[C>0]=1
    C_new=np.zeros_like(C,dtype=float)
    D=np.zeros((im_x,im_y))
    isophotes=isophote(img,0.25)[1]
    # alpha=255

    for point in contour:
        p_x,p_y=point
        temp=0
        for x in range(p_x-patch_size//2,p_x+patch_size//2):
            for y in range(p_y-patch_size//2,p_y+patch_size//2):
                if x<0 or y<0 or x>=im_x or y>=im_y:
                    continue
                temp+=(1/patch_size**2)*C[x,y]
        C_new[p_x,p_y]=temp
        angle_between=np.abs(isophotes[p_x,p_y]-normal[p_x,p_y])
        D[p_x,p_y]=np.abs(np.cos(angle_between))
    ([print(D[x,y],C_new[x,y]) for x,y in contour])
    P=np.array([D[x,y]*C_new[x,y] for x,y in contour])
    return  P


#function  found in matlab documentation
def isophote(L, alpha):
    L = L.astype(float) / 255.0
    theta = np.zeros_like(L)
    Lx, Ly = np.gradient(L)
    I = np.sqrt(Lx**2 + Ly**2)
    I = I / np.max(np.max(I))
    print(np.max(np.max(I)), "\tmax all\t",np.max(I))
    T = (I >= alpha)

    theta[T] = np.arctan2(Ly[T], Lx[T])
    theta_shape = theta.shape

    theta = theta * 180 / np.pi
    I[I < alpha] = 0
    return I, theta


def normal_calc(source_region):
    # Calculate the gradient in the x and y directions
    grad_x = cv2.Sobel(source_region, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(source_region, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the direction of the gradient
    gradient_direction = np.arctan2(grad_y, grad_x)

    normal = (gradient_direction - np.pi / 2)
    return normal

# # Define your continuous function
# def f(x):
#     return x

# # Define the derivative function
# def df(x):
#     return derivative(f, x, dx=1e-8)

