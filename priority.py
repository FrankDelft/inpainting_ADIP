import numpy as np
import cv2
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


def calc_Priority(contour,source_region,patch_size,C,img):
    #iterate through the contour and calculate the patch Priority P
    im_x=img.shape[1]
    im_y=img.shape[0]
    isophotes=isophote(img.copy(),source_region,patch_size,contour) 
    normal=calc_normal(source_region,contour)
    C=calc_Confidence(contour,patch_size,C)
    D=calc_Data(normal,isophotes)
    P_temp=D*C
    P=np.array([P_temp[x,y] for x,y in contour])
    return P,C


def calc_Confidence(contour,patch_size,C):
    #Caclulate the confidence term for patch size
    C_new=C.copy()
    for point in contour:
        p_x,p_y=point
        p_x_min = p_x - patch_size
        p_x_max = p_x + patch_size + 1
        p_y_min = p_y - patch_size  
        p_y_max = p_y + patch_size + 1  
        if p_x_min<0 or p_y_min<0 or p_x_max>=C.shape[0] or p_y_max>=C.shape[1]:
            continue
        C_new[p_x,p_y]=np.sum(np.sum(C[p_x_min:p_x_max,p_y_min:p_y_max]))/(2*patch_size+1)**2
    return C_new


def calc_Data(normal,isophotes):
    D=isophotes*normal
    D=np.sqrt(np.sum(D**2,axis=2))
    return D



#function  found in matlab documentation
def isophote(img,source_region,patch_size,contour):
    #convert image to grayscale
    img=rgb2gray(img)
    img[source_region==-1]=None

    gradient = np.gradient(img)
    gradient_val =np.nan_to_num( np.sqrt(gradient[0]**2 + gradient[1]**2))
    max_gradient=np.zeros([img.shape[0],img.shape[1],2])

    for point in contour:
        patch_x_min = point[0] - patch_size
        patch_x_max = point[0] + patch_size + 1
        patch_y_min = point[1] - patch_size
        patch_y_max = point[1] + patch_size + 1
        patch_y_gradient = gradient[0][patch_x_min:patch_x_max, patch_y_min:patch_y_max]
        patch_x_gradient=gradient[1][patch_x_min:patch_x_max, patch_y_min:patch_y_max]
        patch_grad_val=gradient_val[patch_x_min:patch_x_max, patch_y_min:patch_y_max]


        patch_max_pos = np.unravel_index(
            patch_grad_val.argmax(),
            patch_grad_val.shape
        )

        max_gradient[point[0], point[1], 0] = patch_y_gradient[patch_max_pos]
        max_gradient[point[0], point[1], 1] = patch_x_gradient[patch_max_pos]
        max_gradient = np.nan_to_num(max_gradient)
    return max_gradient


# #function  found in matlab documentation
# def isophote(L,source_region, alpha=0.25):
#     #convert image to grayscale
#     L=cv2.cvtColor(L, cv2.COLOR_RGB2GRAY)
#     #nromalize the image
#     L = L.astype(float)
#     #initialise the gradient and theta
#     Lx, Ly = np.gradient(L)
#     I = np.sqrt(Lx**2 + Ly**2)
#     I = I / np.max(np.max(I))
#     T = (I >= alpha)
#     for x in range(source_region.shape[0]):
#         for y in range(source_region.shape[1]):
#             if source_region[x,y]==-1:
#                 Lx[x,y]=0
#                 Ly[x,y]=0
#     Lx, Ly = Lx / 255, Ly / 255
#     Gradient=np.array([Lx,Ly]).T
#     return Gradient



def calc_normal(source_region,contour_indices):
    kerx = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
    kery = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])
    # Convert source_region to float64
    mask=source_region.copy()
    mask[mask==255]=0
    mask[mask==-1]=1
    
    x_normal = cv2.filter2D(mask.astype(float), -1, kerx)
    y_normal = cv2.filter2D(mask.astype(float), -1, kery)
    normal = np.dstack((x_normal, y_normal))

    height, width = normal.shape[:2]
    norm = np.sqrt(y_normal**2 + x_normal**2) \
                .reshape(height, width, 1) \
                .repeat(2, axis=2)
    norm[norm == 0] = 1

    unit_normal = np.abs(normal/norm)
    return unit_normal



# def calc_normal(source_region,contour_indices):
        
#     # Convert source_region to float64
#     source_region = source_region.astype(np.float64)
#     # Calculate the gradient in the x and y directions
#     grad_x = cv2.Sobel(source_region, cv2.CV_64F, 1, 0, ksize=9)
#     grad_y = cv2.Sobel(source_region, cv2.CV_64F, 0, 1, ksize=9)

#     normal=[]
#     for point in contour_indices:
#         x, y = point
#         N = (grad_y[x,y]**2 + grad_x[x,y]**2)**0.5
#         if N != 0:
#             grad_x[x,y] /= N
#             grad_y[x,y] /= N
#         normal.append([grad_x[x,y],grad_y[x,y]])
#     normal=np.array(normal)
#     return normal
