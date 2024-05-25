import numpy as np
import cv2
def rectangle_target(x_width,y_height,x1,y1,x2,y2):
    image = np.zeros((x_width, y_height), np.uint8) 
    thickness = 1
    contour = cv2.rectangle(image.copy(),[y1,x1],[y2,x2], 255, thickness) 
    # Creating rectangle
    thickness = -1
    target_region = cv2.rectangle(image.copy(),[y1,x1],[y2,x2], 255, thickness) 
    thickness = -1
    source_region =np.array(cv2.bitwise_not(target_region))

    #get x and y pixel co-ordinates
    temp=np.where(np.array(np.array(contour)==255,dtype=int)==1)
    contour_indices = np.array(list(zip(temp[0],temp[1])))
    temp=np.where(np.array(np.array(source_region)==255,dtype=int)==1)
    #source indices
    source_indices = np.array(list(zip(temp[0],temp[1])))
    temp=np.where(np.array(np.array(target_region)==255,dtype=int)==1)
    target_indices = np.array(list(zip(temp[0],temp[1])))
    return contour_indices,source_indices,target_indices
    
def circle_target(x_width,y_height,x1,y1,r):
    image = np.zeros((x_width, y_height), np.uint8) 
    thickness = 1
    contour = cv2.circle(image.copy(),(y1,x1),r, 255, thickness) 
    # Creating rectangle
    thickness = -1
    target_region = cv2.circle(image.copy(),(y1,x1),r, 255, thickness) 
    thickness = -1
    source_region =np.array(cv2.bitwise_not(target_region))

    #get x and y pixel co-ordinates
    temp=np.where(np.array(np.array(contour)==255,dtype=int)==1)
    contour_indices = np.array(list(zip(temp[0],temp[1])))
    temp=np.where(np.array(np.array(source_region)==255,dtype=int)==1)
    #source indices
    source_indices = np.array(list(zip(temp[0],temp[1])))
    temp=np.where(np.array(np.array(target_region)==255,dtype=int)==1)
    target_indices = np.array(list(zip(temp[0],temp[1])))
    return contour_indices,source_indices,target_indices