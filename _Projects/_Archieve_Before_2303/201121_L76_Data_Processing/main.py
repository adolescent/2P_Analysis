# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 12:48:10 2020

@author: ZR
Codes to process L76-201121 Data
"""

from My_Wheels.Translation_Align_Function import Translation_Alignment
import My_Wheels.List_Operation_Kit as List_Tools
#%% First location Align
data_folder=  [r'G:\Test_Data\2P\201121_L76_LM']
run_name = [
    '1-001',
    '1-002',
    '1-003',
    '1-004',
    '1-008',
    ]
loc1_folders = List_Tools.List_Annex(data_folder, run_name)
# Align Run01 Only, usethis run to generate base graph.
Translation_Alignment([loc1_folders[0]],base_mode = 'global',align_range = 20)
# Use 1-001 aligned graph as base to align other graphs.
import cv2 
base = cv2.imread(r'G:\Test_Data\2P\201121_L76_LM\1-001\Base_Loc1-8.tif',-1)
Translation_Alignment([loc1_folders[1]],base_mode = 'input',input_base = base,align_range = 50)
Translation_Alignment([loc1_folders[2]],base_mode = 'input',input_base = base,align_range = 50)
Translation_Alignment([loc1_folders[3]],base_mode = 'input',input_base = base,align_range = 50)
Translation_Alignment([loc1_folders[4]],base_mode = 'input',input_base = base,align_range = 50)
#%% Second Location Align
data_folder = [r'G:\Test_Data\2P\201121_L76_LM']
run_name = ['1-015','1-016']
loc2_folders = List_Tools.List_Annex(data_folder, run_name)
Translation_Alignment(loc2_folders,base_mode = 'global',align_range = 20)
#%%Stim Fram Aligns
from My_Wheels.Stim_Frame_Align import Stim_Frame_Align
import My_Wheels.OS_Tools_Kit as OS_Tools
all_stim_folder = [
    r'G:\Test_Data\2P\201121_L76_LM\201121_L76_stimuli\Run02_2P_G8',
    r'G:\Test_Data\2P\201121_L76_LM\201121_L76_stimuli\Run03_2P_manual_OD8',
    r'G:\Test_Data\2P\201121_L76_LM\201121_L76_stimuli\Run04_2P_RGLum4',
    r'G:\Test_Data\2P\201121_L76_LM\201121_L76_stimuli\Run15_2P_RGLum4'
    ]
for i in range(4):
    _,Frame_Stim_Dic = Stim_Frame_Align(all_stim_folder[i])
    OS_Tools.Save_Variable(all_stim_folder[i], 'Stim_Fram_Align', Frame_Stim_Dic)
#%%Cell Find from Run01 Morphology graph.
from My_Wheels.Cell_Find_From_Graph import Cell_Find_And_Plot
Cell_Find_And_Plot(r'G:\Test_Data\2P\201121_L76_LM\1-001\Results', 'Run_Average_After_Align.tif', 'Morpho',find_thres = 1.5)
#%% Calculate Spike Train of Run01 Morpho cell into each run.
from My_Wheels.Spike_Train_Generator import Spike_Train_Generator
all_run_folder = [
    r'G:\Test_Data\2P\201121_L76_LM\1-002',
    r'G:\Test_Data\2P\201121_L76_LM\1-003',
    r'G:\Test_Data\2P\201121_L76_LM\1-004'
    ]
for i in range(3):
    all_tif_name = OS_Tools.Get_File_Name(all_run_folder[i]+r'\Results\Aligned_Frames')
    cell_information = OS_Tools.Load_Variable(all_run_folder[i]+r'\Results\Morpho\Morpho.cell')['All_Cell_Information']
    stim_train = OS_Tools.Load_Variable(all_run_folder[i]+r'\Results\Stim_Fram_Align.pkl')['Original_Stim_Train']
    F_train,dF_F_train = Spike_Train_Generator(all_tif_name, cell_information,Base_F_type = 'nearest_0',stim_train = stim_train)
    OS_Tools.Save_Variable(all_run_folder[i]+r'\Results\Morpho', 'F_train', F_train)
    OS_Tools.Save_Variable(all_run_folder[i]+r'\Results\Morpho', 'dF_F_train', dF_F_train)
#%% Then Get graph of each run.
from My_Wheels.Standard_Stim_Processor import Standard_Stim_Processor
from My_Wheels.Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
G8_Para = Sub_Dic_Generator('G8+90')
Standard_Stim_Processor(r'G:\Test_Data\2P\201121_L76_LM\1-002',
                        r'G:\Test_Data\2P\201121_L76_LM\1-002\Results\Stim_Fram_Align.pkl',
                        sub_dic = G8_Para,
                        cell_method = r'G:\Test_Data\2P\201121_L76_LM\1-001\Results\Morpho\Morpho.cell',
                        spike_train_path=r'G:\Test_Data\2P\201121_L76_LM\1-002\Results\Morpho\dF_F_train.pkl'
                        )
#%%
OD_Para = Sub_Dic_Generator('OD_2P')
Standard_Stim_Processor(r'G:\Test_Data\2P\201121_L76_LM\1-003',
                        r'G:\Test_Data\2P\201121_L76_LM\1-003\Results\Stim_Fram_Align.pkl',
                        sub_dic = OD_Para,
                        cell_method = r'G:\Test_Data\2P\201121_L76_LM\1-001\Results\Morpho\Morpho.cell',
                        spike_train_path=r'G:\Test_Data\2P\201121_L76_LM\1-003\Results\Morpho\dF_F_train.pkl'
                        )
#%%
RG_Para = Sub_Dic_Generator('RGLum4')
Standard_Stim_Processor(r'G:\Test_Data\2P\201121_L76_LM\1-004',
                        r'G:\Test_Data\2P\201121_L76_LM\1-004\Results\Stim_Fram_Align.pkl',
                        sub_dic = RG_Para,
                        cell_method = r'G:\Test_Data\2P\201121_L76_LM\1-001\Results\Morpho\Morpho.cell',
                        spike_train_path=r'G:\Test_Data\2P\201121_L76_LM\1-004\Results\Morpho\dF_F_train.pkl'
                        )
#%% Find Run15 Cells.
from My_Wheels.Cell_Find_From_Graph import Cell_Find_And_Plot
Cell_Find_And_Plot(r'G:\Test_Data\2P\201121_L76_LM\1-015\Results', 'Run_Average_After_Align.tif', 'Morpho',find_thres = 1.5)
from My_Wheels.Spike_Train_Generator import Spike_Train_Generator
all_run_folder = [
    r'G:\Test_Data\2P\201121_L76_LM\1-015',
    ]
for i in range(1):
    all_tif_name = OS_Tools.Get_File_Name(all_run_folder[i]+r'\Results\Aligned_Frames')
    cell_information = OS_Tools.Load_Variable(all_run_folder[i]+r'\Results\Morpho\Morpho.cell')['All_Cell_Information']
    stim_train = OS_Tools.Load_Variable(all_run_folder[i]+r'\Results\Stim_Fram_Align.pkl')['Original_Stim_Train']
    F_train,dF_F_train = Spike_Train_Generator(all_tif_name, cell_information,Base_F_type = 'nearest_0',stim_train = stim_train)
    OS_Tools.Save_Variable(all_run_folder[i]+r'\Results\Morpho', 'F_train', F_train)
    OS_Tools.Save_Variable(all_run_folder[i]+r'\Results\Morpho', 'dF_F_train', dF_F_train)
#%% Run15 Graphs
RG_Para = Sub_Dic_Generator('RGLum4')
Standard_Stim_Processor(r'G:\Test_Data\2P\201121_L76_LM\1-015',
                        r'G:\Test_Data\2P\201121_L76_LM\1-015\Results\Stim_Fram_Align.pkl',
                        sub_dic = RG_Para,
                        cell_method = r'G:\Test_Data\2P\201121_L76_LM\1-015\Results\Morpho\Morpho.cell',
                        spike_train_path=r'G:\Test_Data\2P\201121_L76_LM\1-015\Results\Morpho\dF_F_train.pkl'
                        )