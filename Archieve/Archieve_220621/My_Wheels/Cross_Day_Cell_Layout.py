# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:17:25 2021

@author: ZR
These functions are used to show cells cross days.
"""
import cv2
from Graph_Matcher import Graph_Matcher
import OS_Tools_Kit as ot
import Graph_Operation_Kit as gt
import numpy as np

def Cross_Day_Cell_Layout(base_dayfolder,
                          target_dayfolder,
                          base_cellnamelist,
                          target_cellnamelist):
    # Step1, do global average and calculate h.
    save_folder = base_dayfolder+r'\_All_Results\Cross_Day'
    ot.mkdir(save_folder)
    base_average_graph = cv2.imread(base_dayfolder+r'\Global_Average.tif',-1)
    target_average_graph = cv2.imread(target_dayfolder+r'\Global_Average.tif',-1)
    merged_graph,_,_,h = Graph_Matcher(base_average_graph, target_average_graph)
    aligned_graph = cv2.warpPerspective(target_average_graph, h, base_average_graph.shape)
    gt.Show_Graph(base_average_graph, 'Base_Graph', save_folder)
    gt.Show_Graph(target_average_graph, 'Target_Graph', save_folder)
    gt.Show_Graph(aligned_graph, 'Aligned_Graph', save_folder)
    gt.Show_Graph(merged_graph, 'Merged_Graph', save_folder)
    # Step2, generate all cell mask alone.
    base_cell_path = ot.Get_File_Name(base_dayfolder,'.ac')[0]
    base_cell = ot.Load_Variable(base_cell_path)
    target_cell_path = ot.Get_File_Name(target_dayfolder,'.ac')[0]
    target_cell = ot.Load_Variable(target_cell_path)
    base_cm = np.zeros(shape = base_average_graph.shape,dtype = 'u2')
    target_cm = np.zeros(shape = target_average_graph.shape,dtype = 'u2')
    for i in range(len(base_cellnamelist)):
        bc_name = base_cellnamelist[i]
        bc_cellinfo = base_cell[bc_name]['Cell_Info']
        y_list,x_list = bc_cellinfo.coords[:,0],bc_cellinfo.coords[:,1]
        base_cm[y_list,x_list] = 65535
    for i in range(len(target_cellnamelist)):
        tc_name = target_cellnamelist[i]
        tc_cellinfo = target_cell[tc_name]['Cell_Info']
        y_list,x_list = tc_cellinfo.coords[:,0],tc_cellinfo.coords[:,1]
        target_cm[y_list,x_list] = 65535
    aligned_target_cm = cv2.warpPerspective(target_cm, h, base_average_graph.shape)
    gt.Show_Graph(base_cm, 'Base_Cell_Mask', save_folder)
    gt.Show_Graph(target_cm, 'Target_Cell_Mask', save_folder)
    gt.Show_Graph(aligned_target_cm, 'Aligned_Target_Cell_Mask', save_folder)
    # Step3,merge graph together   
        #Merge
    Layout_Graph = cv2.cvtColor(base_average_graph,cv2.COLOR_GRAY2RGB).astype('f8')*0.7
    Layout_Graph[:,:,1] += base_cm
    Layout_Graph[:,:,2] += aligned_target_cm
    Layout_Graph = np.clip(Layout_Graph,0,65535).astype('u2')
    gt.Show_Graph(Layout_Graph, 'Cell_Layout_Merge', save_folder)
        #Base
    Base_Cell_Graph = cv2.cvtColor(base_average_graph,cv2.COLOR_GRAY2RGB).astype('f8')*0.7
    Base_Cell_Graph[:,:,1] += base_cm
    Base_Cell_Graph = np.clip(Base_Cell_Graph,0,65535).astype('u2')
    gt.Show_Graph(Base_Cell_Graph, 'Base_Cell_Annotate', save_folder)
        #Target
    Target_Cell_Graph = cv2.cvtColor(base_average_graph,cv2.COLOR_GRAY2RGB).astype('f8')*0.7
    Target_Cell_Graph[:,:,2] += aligned_target_cm
    Target_Cell_Graph = np.clip(Target_Cell_Graph,0,65535).astype('u2')
    gt.Show_Graph(Target_Cell_Graph, 'Target_Cell_Annotate', save_folder)
    return h