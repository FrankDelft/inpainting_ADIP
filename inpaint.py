import numpy as np
import cv2
from priority import calc_Priority, isophote, calc_normal
from util import *
from best_patch import *

import matplotlib.pyplot as plt



class Inpainting:
    def __init__(self,img, source_indices, patch_size=9):

        # intitialisations
        self.im_x = img.shape[1]
        self.im_y = img.shape[0]

        self.patch_size = patch_size

        self.source_region = get_mask(source_indices, self.im_x, self.im_y)

        self.C_init = None
        self.C = None

        self.C_init = self.source_region.copy()
        self.C_init[self.C_init == -1] = 0
        self.C_init[self.C_init == 255] = 1
        self.C_init = self.C_init.astype(float)
        self.C = self.C_init.copy()
        self.source_indices_complete = []

        # lets ensure we only look in the original source region
        for x in range(patch_size, self.im_x - patch_size):
            for y in range(patch_size, self.im_y - patch_size):
                point=(x,y)
                if patch_complete(self.C_init, x, y, patch_size) and point[1] >= self.patch_size and point[1] < self.im_y - self.patch_size -1 and point[0]>= self.patch_size and point[0] < self.im_x- self.patch_size-1:
                    self.source_indices_complete.append((x, y))
        self.source_indices_complete = np.array(self.source_indices_complete)
        self.fill_img = np.ones_like(img) * 255
        for x, y in source_indices:
            self.fill_img[x, y] = img[x, y]
        
        self.source_indices = source_indices
        
    def in_paint_alg(self):

        while np.any(self.source_region == -1):
        
            contour = get_contour(self.source_region.copy())
            # Remove indices within patch_size distance from image edges
            contour = [point for point in contour if point[1] >= self.patch_size and point[1] < self.im_x - self.patch_size-1  and point[0] >= self.patch_size and point[0] < self.im_y - self.patch_size-1]

            P, self.C = calc_Priority(contour,self.source_region.copy(), self.patch_size, self.C, self.fill_img.copy())
            max_index = np.argmax(P)
            p_x, p_y = contour[max_index]

            patch_x_min = p_x - self.patch_size
            patch_x_max = p_x + self.patch_size + 1
            patch_y_min = p_y - self.patch_size
            patch_y_max = p_y + self.patch_size + 1
            max_similarity = patch_distance([patch_x_min, patch_x_max, patch_y_min, patch_y_max], self.fill_img.copy(),
                                            self.patch_size, self.source_indices_complete, self.source_region)
            
            est_x_min = max_similarity[0] - self.patch_size
            est_y_min = max_similarity[1] - self.patch_size
            # Fill the target patch with the source patch and update source region
            source_patch = self.fill_img[est_x_min:est_x_min+self.patch_size*2+1, est_y_min:est_y_min+self.patch_size*2+1]
            target_patch = self.source_region[patch_x_min:patch_x_max, patch_y_min:patch_y_max]
            
            indices_set=np.argwhere(target_patch == -1)

            self.fill_img[patch_x_min:patch_x_max, patch_y_min:patch_y_max][indices_set[:, 0], indices_set[:, 1]] =\
            source_patch[indices_set[:, 0], indices_set[:, 1]]

            self.source_region[patch_x_min:patch_x_max, patch_y_min:patch_y_max] = 255
            self.C[patch_x_min:patch_x_max, patch_y_min:patch_y_max] = self.C[p_x, p_y]
            self.source_indices = np.array(np.where(self.source_region == 255)).T

            # Plot source region and fill image side by side
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(self.source_region, cmap='gray')
            axes[0].set_title('Source Region')
            axes[0].scatter(self.source_indices_complete[:, 0], self.source_indices_complete[:, 1], c='blue', s=5)
            axes[0].scatter(max_similarity[1], max_similarity[0], c='red', s=10)
            axes[1].imshow(self.fill_img)
            axes[1].set_title('Fill Image')

            plt.savefig("source_fill_img.png")
            plt.close(fig)
            plt.clf()

            # Save fill_img to the current directory
            cv2.imwrite("fill_img.jpg", self.fill_img)
