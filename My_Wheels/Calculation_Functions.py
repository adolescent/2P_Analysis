# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:24:36 2020

@author: ZR

Calculation Functions, contains easy matrix calculation tools.
Useful variables contained in there too.
"""
import numpy as np

#%% Function1: Generate 2D Gaussian
def Normalized_2D_Gaussian_Generator(parameter):
    """
    Generate 2D Gaussian matrix(sum = 1), this is useful in graph blur.
    
    Parameters
    ----------
    parameter : (2 element turple)
        Generate parameter.
        parameter[0]:2-element-list. Shape of output matrix
        parameter[1]:sigma. Width of Gaussian function.
    Returns
    -------
    h : (2D ndarray)
        Normalized Gaussian matrix.

    """
    shape = parameter[0]
    sigma = parameter[1]
    m,n = [(ss-1.)/2. for ss in shape] #得到横纵方向的半宽
    y,x = np.ogrid[-m:m+1,-n:n+1]    #左闭右开，得到每一个取值。
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0 #小于精度的置为零。
    sumh = h.sum() # 归一化高斯掩模，使整个模的和为1
    if sumh != 0:
        h /= sumh
    return h
#%% Function 2: Color BGR Dictionary
Color_Dictionary = {}
Color_Dictionary['r'] = (0,0,255)
Color_Dictionary['g'] = (0,255,0)
Color_Dictionary['b'] = (255,0,0)
Color_Dictionary['y'] = (0,255,255)
Color_Dictionary['c'] = (255,255,0)
Color_Dictionary['p'] = (255,0,255)
Color_Dictionary['d'] = (0,0,0)
Color_Dictionary['w'] = (255,255,255)


#%% Function 3 : vector calculator
def Vector_Calculate(point_A,point_B):
    '''
    Calculate vector from A to B. Usually in structure(y,x),use down&right as positive.

    Parameters
    ----------
    point_A : (turple)
        Start point.
    point_B : (turple)
        End point.

    Returns
    -------
    vector : (turple)
        Vector from A to B. Pay attention to positive direction.
    norm : (float)
        Norm of the vector above.

    '''
    y_loc = point_B[0]-point_A[0]
    x_loc = point_B[1]-point_A[1]
    vector = (y_loc,x_loc)
    norm = np.sqrt(np.square(x_loc)+np.square(y_loc))
    return vector,norm

#%% Function 4: 1D Gaussian array generator
def Normalized_1D_Gaussian_Generator(size,sigma):
    '''
    Generate 1D gaussian kernel. This is useful in window averaging.

    Parameters
    ----------
    size : (odd)
        Width of kernel. Must be odd value.
    sigma : (float)
        The std of gaussian array

    Returns
    -------
    gaussian_array : (Array)
        1D Gaussian array in shape given.

    '''
    if size%2 == 0:
        raise ValueError('Gaussian Kernel size must be odd.')
    half_width = (size-1)/2
    x = np.ogrid[-half_width:half_width+1]
    gaussian_array = np.exp(-(x*x)/(2*sigma*sigma))
    gaussian_array[ gaussian_array < np.finfo(gaussian_array.dtype).eps*gaussian_array.max() ] = 0 #小于精度的置为零。
    sumarray = gaussian_array.sum()
    gaussian_array /= sumarray
    return gaussian_array