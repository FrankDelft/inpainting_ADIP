import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2lab
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


def patch_distance(patch_indices, img_rgb, patch_size, source_region,source_indices_complete,euclid):
    max_similarity = [0, 0]
    min_dist = float('inf')
    img_rgb = img_rgb / 255.0
    img_lab = rgb2lab(img_rgb)
    
    # Extract target patch
    target_patch = img_lab[patch_indices[0]:patch_indices[1], patch_indices[2]:patch_indices[3]]
    source_region_patch = source_region[patch_indices[0]:patch_indices[1], patch_indices[2]:patch_indices[3]]
    source_region_patch = np.where(source_region_patch == 255, 1, 0)
   
    target_patch_center = (patch_indices[0] + patch_size, patch_indices[2] + patch_size )
    dist = np.zeros_like(source_region, dtype=float)
    for p_y,p_x in source_indices_complete:
            patch_x_min = p_x - patch_size
            patch_x_max = p_x + patch_size + 1
            patch_y_min = p_y - patch_size
            patch_y_max = p_y + patch_size + 1

            source_patch = img_lab[patch_y_min:patch_y_max, patch_x_min:patch_x_max]
            
            # Apply the mask to the patches
            mask_expanded = source_region_patch[:, :, np.newaxis]
            target_patch_masked = target_patch * mask_expanded
            source_patch_masked = source_patch * mask_expanded
            
            squared_diff = np.sum(np.square(target_patch_masked - source_patch_masked))
            euclidean_dist = np.sqrt((p_x - target_patch_center[1])**2 + (p_y - target_patch_center[0])**2)
            if euclid:
                tot = squared_diff+euclidean_dist
            else:
                 tot = squared_diff
            dist[p_y,p_x]=tot
            if tot < min_dist:
                min_dist = tot
                max_similarity = [p_y, p_x]  
    return max_similarity


def patch_complete(source_region, p_x, p_y, patch_size):
    patch_x_min = p_x - patch_size
    patch_x_max = p_x + patch_size + 1
    patch_y_min = p_y - patch_size
    patch_y_max = p_y + patch_size + 1
    patch=source_region[patch_y_min:patch_y_max, patch_x_min:patch_x_max]
    if np.any(patch != 1)and patch.shape[0]==2*patch_size+1 and patch.shape[1]==2*patch_size+1:
        return False
    return True
