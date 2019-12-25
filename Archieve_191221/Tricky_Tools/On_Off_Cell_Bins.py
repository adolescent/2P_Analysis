# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:30:13 2019

@author: ZR
This function is used to generate Union and Intersection cell graph between different On-Off cell maps
"""

import General_Functions.my_tools as pp
import cv2
import numpy as np

class Union_And_Intersection():
    
    name = r'Ger Union and Intersection cell maps'
    
    def __init__(self,graph_folder):
        
        all_tif_name = pp.file_name(graph_folder,'.tif')
        self.all_cell_graphs = np.zeros(shape = (512,512,len(all_tif_name)),dtype = np.float64)
        for i in range(len(all_tif_name)):
            self.all_cell_graphs[:,:,i] = cv2.imread(all_tif_name[i],0)#0是处理成灰度，-1是保持原样
        
    def Intersection_Graph(self):#做交集图
        intersection_map = np.ones(shape = (512,512),dtype = np.float64)
        for i in range(np.shape(self.all_cell_graphs)[2]):
            intersection_map = intersection_map*self.all_cell_graphs[:,:,i]
        intersection_map = (intersection_map>0)*255#把数值正常化然后*255
        pp.save_graph('Intersection',intersection_map,graph_folder,'.png',8,1)
        
    def Union_Graph(self):
        union_map = np.zeros(shape = (512,512),dtype = np.float64)
        for i in range(np.shape(self.all_cell_graphs)[2]):
            union_map = union_map+self.all_cell_graphs[:,:,i]
        union_map = (union_map>0)*255
        pp.save_graph('Union',union_map,graph_folder,'.png',8,1)
    
    
    
    
    
if __name__ == '__main__':
    
    graph_folder = r'E:\ZR\Data_Temp\190412_L74_LM\0412-On-Off_Maps'
    UAI = Union_And_Intersection(graph_folder)
    UAI.Intersection_Graph()
    UAI.Union_Graph()