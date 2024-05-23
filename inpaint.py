import numpy as np
# from  scipy.misc import derivative
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
def in_paint_alg(img, source_indices,patch_size=9):
    im_x=img.shape[1]
    im_y=img.shape[0]

    C=np.zeros((im_y, im_x), dtype=float)
    D=np.zeros((im_y,im_x))
    cut_img = np.ones_like(img)*-1

    source_region = get_mask(source_indices,im_x,im_y)
    contour=get_contour(source_region)

    # # Plot the contour
    # plt.imshow(source_region, cmap='gray')
    # plt.scatter(contour[:, 1], contour[:, 0], color='red')
    # plt.title('Contour')
    # plt.show()

    normal=calc_normal(source_region,contour)
    isophotes=isophote(img,0.25)[1]
    exit()
    #set pixels that are not in the source region to -1
    for x, y in source_indices:
        cut_img[y, x] = img[y, x]
    
    #iterate through the contour and calculate the patch Priority P
    for point in contour:
        p_x,p_y=point
        temp=0
        #Caclulate the confidence term for patch size   
        for x in range(p_x,p_x+patch_size):
            for y in range(p_y,p_y+patch_size):
                if x<0 or y<0 or x>=im_x or y>=im_y:
                    continue
                temp+=(1/patch_size**2)*source_region[y,x]
        C[p_y,p_x]=temp
        #calculate the isophotes term
        angle_between=np.abs(isophotes[p_y,p_x]-normal[p_y,p_x])
        D[p_y,p_x]=np.abs(np.cos(angle_between))
    P=np.array([D[y,x]*C[y,x] for x,y in contour])

    #now order the patch prioriteis based on the highest priority
    P_zip = list(zip(P, contour))
    P_zip_sorted = sorted(P_zip, key=lambda x: x[0], reverse=True)
    contour_sorted = np.array([x[1] for x in P_zip_sorted])

    
    p_x,p_y=contour_sorted[0]
    patch_x_min = p_x 
    patch_x_max = p_x + patch_size
    patch_y_min = p_y 
    patch_y_max = p_y + patch_size
    #get the patch
    patch=img[patch_y_min:patch_y_max,patch_x_min:patch_x_max]
    #find the most similar patch in the source region
    max_similarity=patch_distance(patch,source_indices,img,patch_size)

    #replace the patch
    est_x_min = max_similarity[0] 
    est_x_max = max_similarity[0] + patch_size 
    est_y_min = max_similarity[1] 
    est_y_max = max_similarity[1] + patch_size 
    cut_img[patch_y_min:patch_y_max,patch_x_min:patch_x_max]=img[est_y_min:est_y_max,est_x_min:est_x_max]

    plt.imshow(cut_img)
    plt.title('Cut Image')
    plt.show()
        
  
def update_indices(cut_img):
    indices = np.array(np.where(cut_img != -1))
    return np.transpose(indices)

def patch_distance(patch,source_indices,img,patch_size):
    max_similarity=[0,0]
    min_dist=100000
    im_x=img.shape[1]
    im_y=img.shape[0]
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    cut_imgx = np.zeros_like(img_lab)

    for x, y in source_indices:
        cut_imgx[y, x] = img_lab[y, x]
    counter=0
    for point in source_indices:
        # counter+=1
        # if counter%100==0:
        #     print(counter)
        p_x,p_y=point
        #check if the patch is within the image
        patch_x_min = p_x 
        patch_x_max = p_x + patch_size
        patch_y_min = p_y 
        patch_y_max = p_y + patch_size
        if patch_x_min<0 or patch_x_max>=im_x or patch_y_min<0 or patch_y_max>=im_y:
            continue
        patch_curr=img_lab[patch_y_min:patch_y_max,patch_x_min:patch_x_max]
        # print(len(x_indices),len(y_indices))
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
 
    theta = theta * 180 / np.pi
    I[I < alpha] = 0

    # plt.imshow(theta, cmap='gray')
    # plt.title('Isophotes')
    # plt.show()
    return I, theta

def get_contour(mask):
    dx = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(dx**2 + dy**2)
    # Define the structuring element
    kernel = np.ones((3,3),np.uint8)
    eroded_magnitude = cv2.erode(magnitude, kernel, iterations=1)
    contour_indices=np.array(np.where(magnitude>0))
    return np.transpose(contour_indices)
    
def get_contour(mask):
    # Convert the mask to uint8
    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    # Define the structuring element
    kernel = np.ones((3,3),np.uint8)

    # Apply erosion
    eroded_mask = cv2.erode(mask_uint8, kernel, iterations = 1)

    # Subtract the eroded image from the original mask to get the contour
    contour_mask = mask_uint8 - eroded_mask

    # Find the indices of the contour pixels
    contour_indices = np.array(np.where(contour_mask > 0))

    return np.transpose(contour_indices)


def get_mask(source_indices,im_x,im_y):
    mask = np.zeros((im_y, im_x), dtype=np.uint8)
    print(mask.shape)
    for x, y in source_indices:
        mask[x, y] = 255
    return mask
    
def calc_normal(source_region,contour_indices):
        
    # Convert source_region to float64
    source_region = source_region.astype(np.float64)
    # Calculate the gradient in the x and y directions
    grad_x = cv2.Sobel(source_region, cv2.CV_64F, 1, 0, ksize=9)
    grad_y = cv2.Sobel(source_region, cv2.CV_64F, 0, 1, ksize=9)

    # Calculate the direction of the gradient
    gradient_direction = np.arctan2(grad_y, grad_x)
    normal = gradient_direction

    # # Plot the source region
    # plt.imshow(source_region, cmap='gray')
    # plt.title('Source Region')

    # # Plot the vectors normal to the source region at the indices of the contour
    # for i in range(0,len(contour_indices),4):
    #     x, y = contour_indices[i,:]
    #     plt.arrow(y, x, np.cos(normal[x, y]), np.sin(normal[x, y]), color='red', width=1)
    # plt.show()

    return normal