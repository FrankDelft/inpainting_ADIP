import numpy as np
import matplotlib.pyplot as plt
import cv2

def patch_distance(patch_indices,img,patch_size,source_indices_complete,source_region):
    max_similarity=[0,0]
    min_dist=10000000000


    t_patch_x_min,t_patch_x_max,t_patch_y_min,t_patch_y_max=patch_indices
    img = img.astype(np.uint8)
    img[img ==-1]=0
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    patch_target=img_lab[t_patch_x_min:t_patch_x_max,t_patch_y_min:t_patch_y_max,:].astype(np.int32)
    source_region_patch=source_region[t_patch_x_min:t_patch_x_max,t_patch_y_min:t_patch_y_max].astype(np.int32)
    for point in source_indices_complete:
        p_y,p_x=point
     
        s_patch_x_min = p_x - patch_size
        s_patch_x_max = p_x + patch_size+1
        s_patch_y_min = p_y - patch_size
        s_patch_y_max = p_y + patch_size+1

        patch_curr=img_lab[s_patch_x_min:s_patch_x_max,s_patch_y_min:s_patch_y_max,:]
        #caclualte the distance between the patches for parts that are in the source region
        distance=0

        mask = source_region_patch == 255

        # print(mask.shape,patch_target.shape,patch_curr.shape)
        estimate_pixels = patch_target[mask]
        source_pixels = patch_curr[mask]
        curr_dist = np.square(estimate_pixels - source_pixels)
        distance = np.sum(curr_dist)

        if distance<min_dist:
            min_dist=distance
            max_similarity[0]=p_x
            max_similarity[1]=p_y
    s_patch_x_min = max_similarity[0] - patch_size
    s_patch_x_max = max_similarity[0] + patch_size+1
    s_patch_y_min = max_similarity[1]- patch_size
    s_patch_y_max = max_similarity[1] + patch_size+1
    patch_curr=img_lab[s_patch_x_min:s_patch_x_max,s_patch_y_min:s_patch_y_max,:]
    # Plot patch_curr and patch_target side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(patch_curr)
    axes[0].set_title('patch_curr')
    axes[1].imshow(patch_target)
    axes[1].set_title('patch_target')
    plt.savefig("patch_comparison.png")
    plt.close(fig)
    plt.clf()
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
