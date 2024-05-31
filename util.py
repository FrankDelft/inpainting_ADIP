import numpy as np
import cv2

def get_contour(mask):
    # Convert the mask to uint8
    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    # Apply Laplace operator to find the contour
    laplacian = cv2.Laplacian(mask_uint8, cv2.CV_8U)

    # Find the indices of the contour pixels
    contour_indices = np.array(np.where(laplacian > 0))
    return np.transpose(contour_indices)


def get_mask(source_indices,im_x,im_y):
    mask = np.ones((im_y, im_x), dtype=np.uint8)*-1
    for x, y in source_indices:
        mask[x, y] = 255
    return mask


def update_indices(cut_img):
    indices = np.array(np.where(cut_img != -1))
    return np.transpose(indices)
