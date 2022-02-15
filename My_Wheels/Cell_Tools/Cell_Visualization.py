# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 14:01:39 2022

@author: ZR
"""

import numpy as np

def Cell_Visualization(cell_names,all_cell_dic,shape = (512,512)):
    
    base_graph = np.zeros(shape = shape,dtype = 'u1')
    for i,cc in enumerate(cell_names):
        c_info = all_cell_dic[cc]['Cell_Info']
        y,x = c_info.coords[:,0],c_info.coords[:,1]
        base_graph[y,x]=255
        
    annotated_graph = base_graph
    return annotated_graph