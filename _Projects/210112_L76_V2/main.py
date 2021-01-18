# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:23:43 2021

@author: ZR
"""

import My_Wheels.List_Operation_Kit as List_Tools
from My_Wheels.Translation_Align_Function import Translation_Alignment
import My_Wheels.OS_Tools_Kit as OS_Tools
import My_Wheels.Graph_Operation_Kit as Graph_Tools
import cv2

day_folder = [r'H:\Test_Data\2P\210112_L76_2P']
run_lists = [
    '1-001',
    '1-007',
    '1-008',
    '1-009',
    '1-011',
    '1-012',
    '1-013',
    '1-014',
    '1-015',
    '1-016'
    ]
all_run_folders = List_Tools.List_Annex(day_folder, run_lists)
#%% Align
Translation_Alignment([all_run_folders[0]])
base_graph = cv2.imread(r'H:\Test_Data\2P\210112_L76_2P\1-001\Results\Run_Average_After_Align.tif',-1)
Translation_Alignment(all_run_folders[1:],base_mode = 'input',input_base = base_graph)
#%% Find cell by global average.
from My_Wheels.Cell_Find_From_Graph import Cell_Find_And_Plot
from Stim_Frame_Align import Stim_Frame_Align
all_stim_cell = Cell_Find_And_Plot(r'E:\Test_Data\2P\210112_L76_2P\1-007\Results', 'Global_Average_After_Align.tif','All_Stim',find_thres = 1.5)
all_stim_folders = [
    r'E:\Test_Data\2P\210112_L76_2P\210112_L76_stimuli\Run07_2P_OD8_auto',
    r'E:\Test_Data\2P\210112_L76_2P\210112_L76_stimuli\Run08_2P_G8',
    r'E:\Test_Data\2P\210112_L76_2P\210112_L76_stimuli\Run09_2P_G8_RF',
    r'E:\Test_Data\2P\210112_L76_2P\210112_L76_stimuli\Run11_2P_G8_RF',
    r'E:\Test_Data\2P\210112_L76_2P\210112_L76_stimuli\Run13_color7_dir8_grating_squarewave_prefsize_BG',
    r'E:\Test_Data\2P\210112_L76_2P\210112_L76_stimuli\Run14_2P_RGLum4_RF',
    r'E:\Test_Data\2P\210112_L76_2P\210112_L76_stimuli\Run15_2P_RGLum4',
    r'E:\Test_Data\2P\210112_L76_2P\210112_L76_stimuli\Run16_shape3_dir8_modified_WJY_201228'
    ]
for i in range(8):
    _,current_stim_dic = Stim_Frame_Align(all_stim_folders[i])
    OS_Tools.Save_Variable(all_stim_folders[i], 'Stim_Frame_Align', current_stim_dic)
#%%Get F and dF trains here.
cell_folder = r'E:\Test_Data\2P\210112_L76_2P\1-007\Results\All_Stim'
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
from Standard_Stim_Processor import One_Key_Stim_Maps
# Run07,OD
OD_Para = Sub_Dic_Generator('OD_2P')
One_Key_Stim_Maps(r'E:\Test_Data\2P\210112_L76_2P\1-007', cell_folder, OD_Para)
# Run08_G8
G8_Para = Sub_Dic_Generator('G8+90')
One_Key_Stim_Maps(r'E:\Test_Data\2P\210112_L76_2P\1-008', cell_folder, G8_Para)
One_Key_Stim_Maps(r'E:\Test_Data\2P\210112_L76_2P\1-009', cell_folder, G8_Para)
One_Key_Stim_Maps(r'E:\Test_Data\2P\210112_L76_2P\1-011', cell_folder, G8_Para)
RG_Para = Sub_Dic_Generator('RGLum4')
One_Key_Stim_Maps(r'E:\Test_Data\2P\210112_L76_2P\1-014', cell_folder, RG_Para)
One_Key_Stim_Maps(r'E:\Test_Data\2P\210112_L76_2P\1-015', cell_folder, RG_Para)
