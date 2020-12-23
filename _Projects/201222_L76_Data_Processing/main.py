# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 13:54:11 2020

@author: ZR
This script is used for 12/22 data processing
"""
import My_Wheels.List_Operation_Kit as List_Tools
import My_Wheels.Graph_Operation_Kit as Graph_Tools
import My_Wheels.OS_Tools_Kit as OS_Tools
import cv2
data_folder = [r'G:\2P\201222_L76_2P']
run_list = [
    '1-001',
    '1-008',
    '1-010',
    '1-011',
    '1-014'
    ]
all_runs = List_Tools.List_Annex(data_folder, run_list)
#%% Add 3 list for run01 to fit ROI change.
run_1 = all_runs[0]
run1_all_tif = OS_Tools.Get_File_Name(run_1)
save_path = run_1+r'\shape_extended'
OS_Tools.mkdir(save_path)
for i in range(len(run1_all_tif)):
    current_graph = cv2.imread(run1_all_tif[i],-1)
    extended_graph = Graph_Tools.Boulder_Extend(current_graph, [0,0,0,3])# 3 pix on the right.
    current_graph_name = run1_all_tif[i].split('\\')[-1]
    Graph_Tools.Show_Graph(extended_graph, current_graph_name, save_path,show_time = 0)
    
    
    
#%%



