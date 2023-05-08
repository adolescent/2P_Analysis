# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 14:01:39 2022

@author: ZR
"""

import numpy as np
import cv2

def Cell_Shape_Visualization(cell_names,all_cell_dic,shape = (512,512)):
    
    base_graph = np.zeros(shape = shape,dtype = 'u1')
    for i,cc in enumerate(cell_names):
        c_info = all_cell_dic[cc]['Cell_Info']
        y,x = c_info.coords[:,0],c_info.coords[:,1]
        base_graph[y,x]=255
        
    annotated_graph = base_graph
    return annotated_graph


def Cell_Weight_Visualization(weights,acd,shape = (512,512)):
    visualized_graph = np.zeros(shape = shape,dtype = 'f8')
    for i,c_weight in enumerate(weights):
        cc_x,cc_y = acd[i+1]['Cell_Loc']
        cc_loc = (acd[i+1]['Cell_Loc'].astype('i4')[1],acd[i+1]['Cell_Loc'].astype('i4')[0])
        visualized_graph = cv2.circle(visualized_graph,cc_loc,4,c_weight,-1)
    return visualized_graph