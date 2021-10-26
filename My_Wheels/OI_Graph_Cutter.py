# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 15:12:10 2021

@author: ZR
This function is used to cut OI graph, comparing with 2P ones.

"""
import Graph_Operation_Kit as Graph_Tools
import OS_Tools_Kit as OS_Tools
import cv2
import skimage.morphology
import numpy as np


def OI_Graph_Cutter(
        OI_Graph_Folder,
        area_mask_path,
        rotate_angle = 90,# clock wise rotation
        OI_Graph_Type = '.bmp'
        ):
    
    
    all_OI_Map_Name = OS_Tools.Get_File_Name(OI_Graph_Folder,file_type = OI_Graph_Type)
    OI_Graph_Num = len(all_OI_Map_Name)
    save_folder = OI_Graph_Folder+r'\Cutted_Graph'
    mask_graph = cv2.imread(area_mask_path,0)# Read in 8bit gray.
    mask_graph = mask_graph>(mask_graph.max()/2)
    mask_graph = skimage.morphology.remove_small_objects(mask_graph,100,connectivity = 1)
    for i in range(OI_Graph_Num):	
        current_OI_graph = cv2.imread(all_OI_Map_Name[i],-1)
        current_graph_name = all_OI_Map_Name[i].split('\\')[-1][:-4]
        non_zero_loc = np.where(mask_graph>0)# unmasked location.
        LU_loc = (non_zero_loc[0].min(),non_zero_loc[1].min())#Left upper graph
        RD_loc = (non_zero_loc[0].max()+1,non_zero_loc[1].max()+1)
        current_masked_OI_graph = current_OI_graph*mask_graph
        #current_masked_OI_graph = current_OI_graph
        cutted_graph = current_masked_OI_graph[LU_loc[0]:RD_loc[0],LU_loc[1]:RD_loc[1]]
        rotated_graph = Graph_Tools.Graph_Twister(cutted_graph, rotate_angle)
        origin_shape = rotated_graph.shape
        resized_graph = cv2.resize(rotated_graph,(origin_shape[0]*7,origin_shape[1]*7))
        Graph_Tools.Show_Graph(rotated_graph, current_graph_name, save_folder,show_time = 0,graph_formation='.png')
        Graph_Tools.Show_Graph(resized_graph,'_Resized_'+current_graph_name, save_folder,show_time = 0,graph_formation='.png')
        #需要再加一个标注在原图中位置的图！
    return True
