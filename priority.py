import numpy as np
import cv2

def calc_Priority(contour,normal,isophotes,patch_size,im_x,im_y,C):
    #iterate through the contour and calculate the patch Priority P
    D=np.zeros((im_y,im_x))
    for point in contour:
        p_x,p_y=point
        temp=0
        #Caclulate the confidence term for patch size   
        for x in range(p_x,p_x+patch_size):
            for y in range(p_y,p_y+patch_size):
                if x<0 or y<0 or x>=im_x or y>=im_y:
                    continue
                temp+=(1/patch_size**2)*C[x,y]
        C[p_x,p_y]=temp
        #calculate the isophotes term
        angle_between=np.abs(isophotes[p_x,p_y]-normal[p_x,p_y])
        D[p_x,p_y]=np.abs(np.cos(angle_between))
    P=np.array([D[x,y]*C[x,y] for x,y in contour])
    return P,D,C


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
