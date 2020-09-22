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

image = cv2.imread(r'D:\ZR\Data_Temp\190412_L74_LM\1-001\results\After_Align_Run001.tif',-1)
#%%
threshold_value = filters.threshold_otsu(image)
labeled_foreground = (image > threshold_value).astype(int)
properties = regionprops(labeled_foreground, image)
center_of_mass = properties[0].centroid
weighted_center_of_mass = properties[0].weighted_centroid
print(center_of_mass)
#%%
annotate_graph = labeled_foreground
y_loc = int(weighted_center_of_mass[0])
x_loc = int(weighted_center_of_mass[1])
annotate_graph[y_loc-5:y_loc+5,x_loc-5:x_loc+5]=2