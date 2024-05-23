import numpy as np
# from  scipy.misc import derivative
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
def in_paint_alg(img, contour, source_indices,patch_size=9):
    im_x=img.shape[1]
    im_y=img.shape[0]
    print("Image shape",img.shape)
    C=source_indices.copy()
    C[C>0]=1
    C_new=np.zeros_like(C,dtype=float)
    D=np.zeros((im_x,im_y))

    #run until no -1 is found in the image
    normal=calc_normal(source_indices,contour,im_x,im_y)
    isophotes=isophote(img,0.25)[1]
    #set pixels that are not in the source region to -1
    cut_img = np.ones_like(img)*-1
    for x, y in source_indices:
        cut_img[y, x] = img[y, x]




def patch_distance(patch,source_region,img,patch_size):
    max_similarity=[0,0]
    min_dist=100000
    im_x=img.shape[1]
    im_y=img.shape[0]
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    for point in source_region:
        p_x,p_y=point
        x_indices = range(p_x - patch_size // 2, p_x + patch_size // 2)
        y_indices = range(p_y - patch_size // 2, p_y + patch_size // 2)
        #ensure all indices are in image bounds
        x_indices = [x for x in x_indices if 0 <= x < im_x]
        y_indices = [y for y in y_indices if 0 <= y < im_y]


        patch_curr=img_lab[y_indices,x_indices]

        distance = np.sum((patch - patch_curr)**2)
      
        if distance<min_dist:
            min_dist=distance
            max_similarity[0]=p_x
            max_similarity[1]=p_y
    return max_similarity



#function  found in matlab documentation
def isophote(L, alpha):
    #convert image to grayscale
    L=cv2.cvtColor(L, cv2.COLOR_RGB2GRAY)
    #nromalize the image
    L = L.astype(float) / 255.0
    #initialise the gradient and theta
    theta = np.zeros_like(L)
    Lx, Ly = np.gradient(L)
    I = np.sqrt(Lx**2 + Ly**2)
    I = I / np.max(np.max(I))
    T = (I >= alpha)

    theta[T] = np.arctan2(Ly[T], Lx[T])
    theta_shape = theta.shape

    theta = theta * 180 / np.pi
    I[I < alpha] = 0
    return I, theta




def calc_normal(source_indices,contour_indices,im_x,im_y):

    source_region = np.zeros((im_y, im_x), dtype=int) 
    source_region[source_indices[:, 1], source_indices[:, 0]] = 255
    
    # Convert source_region to float64
    source_region = source_region.astype(np.float64)
    # Calculate the gradient in the x and y directions
    grad_x = cv2.Sobel(source_region, cv2.CV_64F, 1, 0, ksize=9)
    grad_y = cv2.Sobel(source_region, cv2.CV_64F, 0, 1, ksize=9)

    # Calculate the direction of the gradient
    gradient_direction = np.arctan2(grad_y, grad_x)
    non_zero_indices = np.nonzero(gradient_direction)

    normal = gradient_direction

    # # Plot the source region
    # plt.imshow(source_region, cmap='gray')
    # plt.title('Source Region')

    # # Plot the vectors normal to the source region at the indices of the contour
    # for i in range(0,len(contour_indices)):
    #     x, y = contour_indices[i]
    #     plt.arrow(x, y, np.cos(normal[y, x]), np.sin(normal[y, x]), color='red', width=1)
    # plt.show()
    
    return normal