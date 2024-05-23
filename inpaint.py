import numpy as np
# from  scipy.misc import derivative
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
def in_paint_alg(img, contour, source_indices,patch_size=9):
    im_x=img.shape[1]
    im_y=img.shape[0]
    # C=source_indices.copy()
    # C[C>0]=1
    C=np.zeros((im_y, im_x), dtype=float)
    D=np.zeros((im_x,im_y))
    cut_img = np.ones_like(img)*-1
    source_region = np.zeros((im_y, im_x), dtype=int)

    #run until no -1 is found in the image
    normal=calc_normal(source_indices,contour,im_x,im_y)
    isophotes=isophote(img,0.25)[1]

    #set pixels that are not in the source region to -1
    for x, y in source_indices:
        cut_img[y, x] = img[y, x]
        source_region[y, x] = 1
    
    #iterate through the contour and calculate the patch Priority P
    for point in contour:
        p_x,p_y=point
        temp=0
        #Caclulate the confidence term for patch size   
        for x in range(p_x-patch_size//2,p_x+patch_size//2):
            for y in range(p_y-patch_size//2,p_y+patch_size//2):
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

    for point in contour_sorted:
        p_x,p_y=point
        #get the patch
        patch=img[p_y-patch_size//2:p_y+patch_size//2,p_x-patch_size//2:p_x+patch_size//2]
        #find the most similar patch in the source region
        max_similarity=patch_distance(patch,source_indices,img,patch_size)
        #replace the patch
        cut_img[p_y-patch_size//2:p_y+patch_size//2,p_x-patch_size//2:p_x+patch_size//2]=img[max_similarity[1]-patch_size//2:max_similarity[1]+patch_size//2,max_similarity[0]-patch_size//2:max_similarity[0]+patch_size//2]
  


def update_indices(cut_img):
    im_x = cut_img.shape[1]
    im_y = cut_img.shape[0]
    indices = np.array(np.where(cut_img != -1))
    return np.transpose(indices)

def patch_distance(patch,source_indices,img,patch_size):
    max_similarity=[0,0]
    min_dist=100000
    im_x=img.shape[1]
    im_y=img.shape[0]
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    cut_img = np.zeros_like(img_lab)
    print("img_lab",img_lab.shape,"img",img.shape)
    for x, y in source_indices:
        cut_img[y, x] = img_lab[y, x]
    for point in source_indices:
        p_x,p_y=point
        x_indices = range(p_x - patch_size // 2, p_x + patch_size // 2)
        y_indices = range(p_y - patch_size // 2, p_y + patch_size // 2)
        #check if the patch is within the image
        if p_x-patch_size // 2<0 or p_x+patch_size // 2>=im_x or p_y-patch_size // 2<0 or p_y+patch_size // 2>=im_y:
            continue
        patch_curr=img_lab[y_indices,x_indices]
        print(patch.shape,patch_curr.shape)
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