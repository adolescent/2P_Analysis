# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:07:16 2019

@author: ZR
"""

import cv2
import numpy as np
import functions_OD as pp
from scipy import stats
import random
class Graph_Generation():
    
    name = 'Generate functional map'
    
    def __init__(self,stim_set_A,stim_set_B,map_name,save_folder):
        
        self.map_folder = save_folder+r'\\Stim_Graphs'
        pp.mkdir(self.map_folder)
        self.stim_set_A = stim_set_A
        self.stim_set_B = stim_set_B
        self.map_name = map_name
    
    def ID_Configuration(self):
        
        Frame_Stim_Check = pp.read_variable(save_folder+r'\\Frame_Stim_Check.pkl')
        self.frame_set_A = []
        self.frame_set_B = []
        for i in range(len(self.stim_set_A)):
            self.frame_set_A.extend(Frame_Stim_Check[self.stim_set_A[i]])
        for i in range(len(self.stim_set_B)):
            self.frame_set_B.extend(Frame_Stim_Check[self.stim_set_B[i]])
            
    def Sub_Map(self):
        
        aligned_tif_name = pp.read_variable(save_folder+r'\\aligned_tif_name.pkl')
        average_frame_A = np.zeros(shape = (512,512),dtype = np.float64)
        average_frame_B = np.zeros(shape = (512,512),dtype = np.float64)
        for i in range(0,len(self.frame_set_A)):#得到A的平均图
            temp_frame = np.float64(cv2.imread(aligned_tif_name[self.frame_set_A[i]],-1))
            average_frame_A = average_frame_A + temp_frame/len(self.frame_set_A)
        for i in range(0,len(self.frame_set_B)):#得到B的平均图
            temp_frame = np.float64(cv2.imread(aligned_tif_name[self.frame_set_B[i]],-1))
            average_frame_B = average_frame_B + temp_frame/len(self.frame_set_B)
        #接下来做减图和clip
        sub_graph = average_frame_A - average_frame_B
        clip_min = sub_graph.mean()-3*sub_graph.std()
        clip_max = sub_graph.mean()+3*sub_graph.std()
        sub_graph_clipped = np.clip(sub_graph,clip_min,clip_max)#对减图进行最大和最小值的clip
        norm_sub_graph = (sub_graph_clipped-sub_graph_clipped.min())/(sub_graph_clipped.max()-sub_graph_clipped.min())
        #以上得到了clip且归一化了的map
        
        
        
        
        
if __name__ =='__main__':
    save_folder = r'E:\ZR\Data_Temp\190412_L74_LM\1-001\results'
    set_A = []#这里画图画的是A-B
    set_B = []
    GG = 