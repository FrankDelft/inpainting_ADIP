import numpy as np
# from  scipy.misc import derivative
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from priority import calc_Priority,isophote,calc_normal
from util import *


def in_paint_alg(img, source_indices,patch_size=9):
    im_x=img.shape[1]
    im_y=img.shape[0]
    isophotes=isophote(img,0.25)[1]
    source_region = get_mask(source_indices,im_x,im_y)
    C_init=source_region.copy()
    C_init[C_init ==-1]=0
    C_init[C_init ==255]=1
    C_init = C_init.astype(float)
    C=C_init.copy()

    source_indices_complete=[]
    #lets ensuure we only lookin the orignal source region
    for x in range(patch_size,im_x-patch_size):
        for y in range(patch_size,im_y-patch_size):
            if patch_complete(C_init,x,y,patch_size):
                source_indices_complete.append((x,y))
    
    fill_img = np.ones_like(img)*-1
    for x, y in source_indices:
        fill_img[x, y] = img[x, y]
    masked_img = fill_img.copy()
    masked_img[masked_img == -1] = 0
    
    while np.any(source_region == -1):

        contour=get_contour(source_region)
        normal=calc_normal(source_region,contour)
        
        P,D,C=calc_Priority(contour,normal,isophotes,patch_size,im_x,im_y,C)
        # plot C
     
        #now order the patch prioriteis based on the highest priority
        P_zip = list(zip(P, contour))
        P_zip_sorted = sorted(P_zip, key=lambda x: x[0], reverse=True)
        contour_sorted = np.array([x[1] for x in P_zip_sorted])

        
        p_x,p_y=contour_sorted[0]
        patch_x_min = p_x - patch_size
        patch_x_max = p_x + patch_size+1
        patch_y_min = p_y - patch_size
        patch_y_max = p_y + patch_size+1
        #get the patch
        patch=fill_img[patch_x_min:patch_x_max,patch_y_min:patch_y_max]
        #find the most similar patch in the source region
        max_similarity=patch_distance(patch,fill_img.copy(),patch_size,source_indices_complete)
        #replace the patch
        est_x_min = max_similarity[0] - patch_size
        est_x_max = max_similarity[0] + patch_size+1
        est_y_min = max_similarity[1] - patch_size 
        est_y_max = max_similarity[1] + patch_size+1
        fill_img[patch_x_min:patch_x_max,patch_y_min:patch_y_max]=masked_img[est_x_min:est_x_max,est_y_min:est_y_max]
        #update the source region and indices
        source_region = np.array(fill_img[:,:,0])
        source_region[source_region !=-1]=255
        source_region[source_region !=-1]=255
        source_indices=np.array(np.where(source_region==255)).T
        

        # Save fill_img to the current directory
        cv2.imwrite("fill_img.jpg", fill_img)
        # plt.imshow(source_region, cmap='gray')
        # plt.title('Source Region')
        # plt.show()
        



def patch_distance(patch,img,patch_size,source_indices_complete):
    max_similarity=[0,0]
    min_dist=100000000
    im_x=img.shape[1]
    im_y=img.shape[0]

    img = img.astype(np.uint8)
    img[img ==-1]=0
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    source_indices_complete = np.array(source_indices_complete)
    for point in source_indices_complete:
        p_x,p_y=point
        #check if the patch is within the image
        patch_x_min = p_x - patch_size
        patch_x_max = p_x + patch_size+1
        patch_y_min = p_y - patch_size
        patch_y_max = p_y + patch_size+1
        if patch_x_min<0 or patch_x_max>=im_y or patch_y_min<0 or patch_y_max>=im_x-1:
            continue
 
        patch_curr=img_lab[patch_x_min:patch_x_max,patch_y_min:patch_y_max,:]
        distance = np.sum((patch - patch_curr)**2)
        if distance<min_dist:
            min_dist=distance
            max_similarity[0]=p_x
            max_similarity[1]=p_y
    
    return max_similarity



def patch_complete(C_original,p_x,p_y,patch_size):
    patch_x_min = p_x - patch_size
    patch_x_max = p_x + patch_size+1
    patch_y_min = p_y - patch_size
    patch_y_max = p_y + patch_size+1
    # print(patch_x_min,patch_x_max,patch_y_min,patch_y_max, "\t", C_original.shape)
    for x in range(patch_x_min,patch_x_max):
        for y in range(patch_y_min,patch_y_max):
            if C_original[y,x]!=1:
                return False
    return True
