# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 09:57:51 2021

@author: adolescent
"""
from My_Wheels.Translation_Align_Function import Translation_Alignment
import My_Wheels.List_Operation_Kit as List_Tools
import cv2
from My_Wheels.Stim_Frame_Align import Stim_Frame_Align
import My_Wheels.OS_Tools_Kit as OS_Tools

data_folder = [r'E:\Test_Data\2P\210101_L76_2P']
spon_run_folders = ['1-001','1-002']
all_spon_folders = List_Tools.List_Annex(data_folder, spon_run_folders)
Translation_Alignment(all_spon_folders,graph_shape=(376,352))
#%% Then align all other spon runs to Run01/Run02
after_spons = [r'E:\Test_Data\2P\210101_L76_2P\1-010',
               r'E:\Test_Data\2P\210101_L76_2P\1-018']
base_graph = cv2.imread(r'E:\Test_Data\2P\210101_L76_2P\1-002\Results\Global_Average_After_Align.tif',-1)
Translation_Alignment(after_spons,base_mode = 'input',input_base = base_graph,graph_shape = (376,352))
#%% Then analyze full frame stim maps.
basic_stim_folders = [r'I:\Test_Data\2P\210101_L76_2P\1-014',
                      r'I:\Test_Data\2P\210101_L76_2P\1-016',
                      r'I:\Test_Data\2P\210101_L76_2P\1-017'
                      ]
Translation_Alignment(basic_stim_folders,base_mode = 0)
#Get stim frame aligns
all_stim_folders = [r'I:\Test_Data\2P\210101_L76_2P\210101_L76_2P_stimuli\Run14_2P_OD8_auto',
                    r'I:\Test_Data\2P\210101_L76_2P\210101_L76_2P_stimuli\Run16_2P_G8',
                    r'I:\Test_Data\2P\210101_L76_2P\210101_L76_2P_stimuli\Run17_2P_RGLum4'
                    ]
for i in range(3):
    _,Stim_Frame_Dic = Stim_Frame_Align(all_stim_folders[i])
    OS_Tools.Save_Variable(all_stim_folders[i], 'Stim_Frame_Align', Stim_Frame_Dic)
#%% Then cell find for all stim maps.
from My_Wheels.Cell_Find_From_Graph import Cell_Find_And_Plot
Cell_Find_And_Plot(r'I:\Test_Data\2P\210101_L76_2P\1-014\Results','Global_Average_After_Align.tif','Stim_Global',find_thres = 1.5) 
cell_folder = r'I:\Test_Data\2P\210101_L76_2P\1-014\Results\Stim_Global'
#%%Then calculate all stim graphs.
from My_Wheels.Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
from My_Wheels.Standard_Stim_Processor import One_Key_Stim_Maps
OD_para = Sub_Dic_Generator('OD_2P')
One_Key_Stim_Maps(r'I:\Test_Data\2P\210101_L76_2P\1-014', cell_folder, OD_para)



