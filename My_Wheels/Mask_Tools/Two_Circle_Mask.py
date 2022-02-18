# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:44:23 2022

@author: adolescent
"""
import numpy as np
import random
import cv2

def Two_Circle_Mask(radius = 70,dist = 200,shape = (512,512),time_lim = 100000):
    '''
    Generate 2 circle mask have specific dist & radius.

    Parameters
    ----------
    radius : (int), optional
        Radius of circle mask. The default is 70.
    dist : (int), optional
        Distance between 2 circles. The default is 200.
    shape : (2-element-turple), optional
        Shape of output graph. The default is (512,512).

    Returns
    -------
    mask_A : (2D Array)
        Circle mask A.
    mask_B : (2D Array)
        Circle mask B.
    all_mask : (2D Array)
        Mask of A and B. color map.

    '''
    
    # get point B
    legal_B = False
    times = 0
    while(legal_B == False):
        # get point A
        point_A = (random.randint(radius,shape[0]-radius),random.randint(radius,shape[1]-radius))
        angle = random.uniform(0, np.pi*2)
        vector = (int(dist*np.sin(angle)),int(dist*np.cos(angle)))
        point_B = (point_A[0]+vector[0],point_A[1]+vector[1])
        if point_B[0]>radius and point_B[0]<(shape[0]-radius):
            if point_B[1]>radius and point_B[1]<(shape[1]-radius):
                legal_B = True
        times+=1
        if times >time_lim:
            raise IOError('Invalid mask!')
        
        
    mask_A = np.zeros(shape = shape,dtype = 'u1')
    mask_B = np.zeros(shape = shape,dtype = 'u1')
    mask_A = cv2.circle(mask_A,(point_A[1],point_A[0]),radius,(255),-1) # cv2 read XY sequence..
    mask_B = cv2.circle(mask_B,(point_B[1],point_B[0]),radius,(255),-1) # cv2 read XY sequence..
    # Add mask together.
    all_mask = np.zeros(shape = (shape[0],shape[1],3),dtype = 'u1')
    all_mask[:,:,1] = mask_A
    all_mask[:,:,2] = mask_B
    
    return mask_A,mask_B,all_mask