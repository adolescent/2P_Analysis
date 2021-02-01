# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:38:19 2021

@author: adolescent

"""
import My_Wheels.OS_Tools_Kit as OS_Tools
from My_Wheels.Translation_Align_Function import Translation_Alignment
import My_Wheels.List_Operation_Kit as List_Tools

data_folder = [r'G:\Test_Data\2P\210123_L76_2P']
all_useful_run =[
    '1-001',
    '1-008',
    '1-009',
    '1-011',
    '1-012',
    '1-013',
    '1-014',
    '1-015'
    ]
#Remenber,Spon runs have ROI.
all_folder_name = List_Tools.List_Annex(data_folder, all_useful_run)
spon_folders = List_Tools.List_Slicer(all_folder_name, [0,1])
# Align Spon runs
Translation_Alignment(spon_folders,base_mode=0,graph_shape = (318,316))
# Then align all stim runs.
Translation_Alignment(all_folder_name[2:],base_mode=0,graph_shape = (512,512))

#%% Cell find
from My_Wheels.Cell_Find_From_Graph import Cell_Find_And_Plot
spon_cells = Cell_Find_And_Plot(r'G:\Test_Data\2P\210123_L76_2P\1-001\Results', 'Global_Average_After_Align.tif', 'Spon_ROIs',find_thres = 1.5)
stim_cells = Cell_Find_And_Plot(r'G:\Test_Data\2P\210123_L76_2P\1-009\Results', 'Global_Average_After_Align.tif', 'All_Stim_Cell',find_thres = 1.5)
#%% Then do stim fram align
from My_Wheels.Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'G:\Test_Data\2P\210123_L76_2P\210123_L76_2P_stimuli')
#%% At last, do data processing.
from My_Wheels.Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
from My_Wheels.Standard_Stim_Processor import One_Key_Stim_Maps
OD_Para = Sub_Dic_Generator('OD_2P')
cell_folder = r'G:\Test_Data\2P\210123_L76_2P\1-009\Results\All_Stim_Cell'
One_Key_Stim_Maps(r'G:\Test_Data\2P\210123_L76_2P\1-009', cell_folder, OD_Para)
