# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:24:36 2020

@author: ZR

Calculation Functions, contains easy matrix calculation tools.
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
    m,n = [(ss-1.)/2. for ss in shape] #得到横纵方向的半高宽
    y,x = np.ogrid[-m:m+1,-n:n+1]    #左闭右开，得到每一个取值。
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0 #小于精度的置为零。
    sumh = h.sum() # 归一化高斯掩模，使整个模的和为1
    if sumh != 0:
        h /= sumh
    return h