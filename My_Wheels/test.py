# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:25:07 2020

@author: ZR
Test Run File
Do not save.
"""
#%%
import cv2
from skimage import filters
from skimage.color import rgb2gray  # only needed for incorrectly saved images
from skimage.measure import regionprops

#image = cv2.imread(r'D:\ZR\Data_Temp\190412_L74_LM\1-001\results\After_Align_Run001.tif',-1)
#%%
import numpy as np
from matplotlib import pyplot as plt 
fig = plt.figure()
ax = plt.subplot()
ax.scatter(20,30, alpha=1,s= 10)
ax.scatter(25,35	, alpha=1,s= 10)
ax.scatter((25,30),alpha = 1,s = 10)
