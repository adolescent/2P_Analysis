# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 10:35:15 2020

@author: ZR

This function is used to define ROI region of the whole graph.
"""
from My_Wheels.Affine_Alignment import Affine_Core_Point_Equal
import My_Wheels.Graph_Operation_Kit as Graph_Tools
import numpy as np
import cv2

def Graph_Matcher(
        full_graph,
        ROI_graph,
        ):
    if full_graph.dtype == 'u2':
        full_graph = (full_graph//256).astype('u1')
    aligned_ROI,h = Affine_Core_Point_Equal(ROI_graph,full_graph,targ_gain = 1)
    aligned_ROI = (aligned_ROI//256).astype('u1')
    # Here we get affine matrix h and aligned ROI graph.
    merged_graph = cv2.cvtColor(full_graph,cv2.COLOR_GRAY2RGB).astype('f8')
    location_graph = cv2.cvtColor(full_graph,cv2.COLOR_GRAY2RGB).astype('f8') # Define location here.
    merged_graph[:,:,1] += aligned_ROI
    merged_graph = np.clip(merged_graph,0,255).astype('u1')
    # Then we annotate graph boulder in graph.
    rectangle = np.zeros(np.shape(ROI_graph),dtype = 'u1')
    rectangle = Graph_Tools.Boulder_Fill(rectangle, [3,3,3,3], 255)
    ROI_boulder = cv2.warpPerspective(rectangle, h, (512,512))
    location_graph[:,:,2] += ROI_boulder
    location_graph = np.clip(location_graph,0,255).astype('u1')
    return merged_graph,location_graph,ROI_boulder