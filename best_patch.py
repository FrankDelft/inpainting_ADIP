import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray, rgb2lab
# def patch_distance(patch_indices,img,patch_size,source_indices_complete,source_region):
#     max_similarity=[0,0]
#     min_dist=10000000000

#     t_patch_x_min,t_patch_x_max,t_patch_y_min,t_patch_y_max=patch_indices
#     img = img.astype(np.uint8)
#     img[img ==-1]=0
#     img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
#     dist=np.zeros_like(source_region)
#     patch_target=img_lab[t_patch_x_min:t_patch_x_max,t_patch_y_min:t_patch_y_max,:].astype(np.int32)
#     source_region_patch=source_region[t_patch_x_min:t_patch_x_max,t_patch_y_min:t_patch_y_max].astype(np.int32)
#     for point in source_indices_complete:
#         p_y,p_x=point
     
#         s_patch_x_min = p_x - patch_size
#         s_patch_x_max = p_x + patch_size+1
#         s_patch_y_min = p_y - patch_size
#         s_patch_y_max = p_y + patch_size+1

#         patch_curr=img_lab[s_patch_x_min:s_patch_x_max,s_patch_y_min:s_patch_y_max,:]
#         #caclualte the distance between the patches for parts that are in the source region
#         distance=0
#         mask = source_region_patch == 255
#         estimate_pixels = patch_target[mask]
#         source_pixels = patch_curr[mask]
#         curr_dist = np.square(estimate_pixels - source_pixels)
#         distance = np.abs(np.sum(curr_dist))
#         dist[p_x,p_y]=distance
#         if distance<=min_dist:
#             min_dist=distance
#             max_similarity[0]=p_x
#             max_similarity[1]=p_y

#     s_patch_x_min = max_similarity[0] - patch_size
#     s_patch_x_max = max_similarity[0] + patch_size+1
#     s_patch_y_min = max_similarity[1]- patch_size
#     s_patch_y_max = max_similarity[1] + patch_size+1
#     patch_curr=img_lab[s_patch_x_min:s_patch_x_max,s_patch_y_min:s_patch_y_max,:]
#     # fig, axs = plt.subplots(1, 3, figsize=(10, 4))

#     # axs[0].imshow(patch_target)
#     # axs[0].set_title('Target Patch')

#     # axs[1].imshow(source_region_patch)
#     # axs[1].set_title('Source Patch')

#     # axs[2].imshow(patch_curr)
#     # axs[2].set_title('Current Patch')

#     # plt.show()
#     return max_similarity


# def patch_distance(patch_indices,img_rgb,patch_size,source_indices_complete,source_region):
#     max_similarity=[0,0]
#     min_dist=10000000000
#     img_lab = rgb2lab(img_rgb)
#     target_patch = img_lab[patch_indices[0]:patch_indices[1], patch_indices[2]:patch_indices[3]]
#     for x,y in source_indices_complete:
#         p_x, p_y = x,y
#         source_patch = img_lab[p_y - patch_size:p_y + patch_size + 1, p_x - patch_size:p_x + patch_size + 1]
#         mask = source_region[p_y - patch_size:p_y + patch_size + 1, p_x - patch_size:p_x + patch_size + 1] == 255
#         rgb_mask=mask.reshape(patch_size*2+1,patch_size*2+1,1).repeat(3,axis=2)
#         if np.any(mask==False):
#             continue
#         target_data=target_patch*rgb_mask
#         source_data=source_patch*rgb_mask
#         ssd=np.sum((target_data-source_data)**2)
#         euclidean_dist=np.sqrt(patch_indices[0]-p_x-patch_size)**2+(patch_indices[2]-p_y-patch_size)**2
#         dist=ssd+euclidean_dist
#         if dist<min_dist:
#             min_dist=dist
#             max_similarity[0]=p_y
#             max_similarity[1]=p_x
#         return max_similarity



    

def patch_distance(patch_indices,img_rgb,patch_size,source_indices_complete,source_region):
    max_similarity=[0,0]
    min_dist=10000000000
    # img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    img_lab = rgb2lab(img_rgb)
    target_patch = img_lab[patch_indices[0]:patch_indices[1], patch_indices[2]:patch_indices[3]]
    source_region_patch=source_region[patch_indices[0]:patch_indices[1], patch_indices[2]:patch_indices[3]]

    # patch_indices=np.argwhere(source_region_patch==source_region_patch)
    for point in source_indices_complete:
        p_x, p_y = point
        source_patch = img_lab[p_y - patch_size:p_y + patch_size + 1, p_x - patch_size:p_x + patch_size + 1]
        mask = source_region_patch == 255
        difference = source_patch.astype(float) - target_patch.astype(float)
        euclidean_dist=np.sqrt((p_x-(patch_indices[0]+patch_size))**2+(p_y-(patch_indices[2]+patch_size))**2)
        sum_difference = np.abs(np.sum(np.square(difference[mask])))-euclidean_dist
        
        if sum_difference<min_dist:
            min_dist=sum_difference
            max_similarity[0]=p_y
            max_similarity[1]=p_x
    return max_similarity

# def patch_distance(patch_indices,img,patch_size,source_indices_complete,source_region):
#     max_similarity=[0,0]
#     min_dist=10000000000


#     t_patch_x_min,t_patch_x_max,t_patch_y_min,t_patch_y_max=patch_indices
#     img = img.astype(np.uint8)
#     img[img ==-1]=0
#     img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

#     patch_target=img_lab[t_patch_x_min:t_patch_x_max,t_patch_y_min:t_patch_y_max,:].astype(np.int32)
#     source_region_patch=source_region[t_patch_x_min:t_patch_x_max,t_patch_y_min:t_patch_y_max].astype(np.int32)
#     for point in source_indices_complete:
#         p_y,p_x=point
     
#         s_patch_x_min = p_x - patch_size
#         s_patch_x_max = p_x + patch_size+1
#         s_patch_y_min = p_y - patch_size
#         s_patch_y_max = p_y + patch_size+1

#         patch_curr=img_lab[s_patch_x_min:s_patch_x_max,s_patch_y_min:s_patch_y_max,:]
#         #caclualte the distance between the patches for parts that are in the source region
#         distance=0

#         mask = source_region_patch == 255

#         # print(mask.shape,patch_target.shape,patch_curr.shape)
#         estimate_pixels = patch_target[mask]
#         source_pixels = patch_curr[mask]
#         curr_dist = np.square(estimate_pixels - source_pixels)
#         distance = np.sum(curr_dist)

#         if distance<min_dist:
#             min_dist=distance
#             max_similarity[0]=p_x
#             max_similarity[1]=p_y
#     s_patch_x_min = max_similarity[0] - patch_size
#     s_patch_x_max = max_similarity[0] + patch_size+1
#     s_patch_y_min = max_similarity[1]- patch_size
#     s_patch_y_max = max_similarity[1] + patch_size+1
#     patch_curr=img_lab[s_patch_x_min:s_patch_x_max,s_patch_y_min:s_patch_y_max,:]

#     return max_similarity

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
