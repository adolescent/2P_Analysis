# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:08:38 2020

@author: ZR
"""

import My_Wheels.Translation_Align_Function as Align
import My_Wheels.List_Operation_Kit as List_Tools
import My_Wheels.OS_Tools_Kit as OS_Tools
import numpy as np
#%% Align First
data_folder = [r'G:\Test_Data\2P\201211_L76_2P']
run_folder = [
    '1-001',
    '1-010',
    '1-012',
    '1-013',
    '1-014'
    ]
all_run_folder = List_Tools.List_Annex(data_folder, run_folder)
Align.Translation_Alignment(all_run_folder,base_mode = 1,align_range=50,align_boulder=50,big_memory_mode=True)
#%% Then find cell from after align spon graph.
from My_Wheels.Cell_Find_From_Graph import Cell_Find_And_Plot
Cell_Find_And_Plot(r'G:\Test_Data\2P\201211_L76_2P\1-001\Results', 'Global_Average_After_Align.tif', 'Global_Morpho',find_thres= 1.5,shape_boulder = [20,20,30,20])
#%% Then calculate the stim train of each stim series.
from My_Wheels.Stim_Frame_Align import Stim_Frame_Align
all_stim_folder = [
    r'G:\Test_Data\2P\201211_L76_2P\201211_L76_2P_stimuli\Run10_2P_G8',
    r'G:\Test_Data\2P\201211_L76_2P\201211_L76_2P_stimuli\Run12_2P_OD8_auto',
    r'G:\Test_Data\2P\201211_L76_2P\201211_L76_2P_stimuli\Run14_2P_RGLum4',
    ]
for i in range(3):
    _,current_stim_dic = Stim_Frame_Align(all_stim_folder[i])
    OS_Tools.Save_Variable(all_stim_folder[i], 'Stim_Frame_Align', current_stim_dic)
#%% Then calculate spike train of different runs.
from My_Wheels.Spike_Train_Generator import Spike_Train_Generator
#Cycle basic stim map. this maps have 
for i,index in enumerate([1,2,4]):
    current_aligned_tif_name  = OS_Tools.Get_File_Name(all_run_folder[index]+r'\Results\Aligned_Frames')
    current_stim = OS_Tools.Load_Variable(all_stim_folder[i],file_name='Stim_Frame_Align.pkl')['Original_Stim_Train']
    current_cell_info = OS_Tools.Load_Variable(all_run_folder[index]+r'\Results\Global_Morpho\Global_Morpho.cell')['All_Cell_Information']
    F_train,dF_F_train = Spike_Train_Generator(current_aligned_tif_name, current_cell_info,Base_F_type= 'nearest_0',stim_train = current_stim)
    OS_Tools.Save_Variable(all_run_folder[index]+r'\Results', 'F_train', F_train)
    OS_Tools.Save_Variable(all_run_folder[index]+r'\Results', 'dF_F_train', dF_F_train)

#%% Then calculate standard stim map.
from My_Wheels.Standard_Stim_Processor import Standard_Stim_Processor
from My_Wheels.Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
Standard_Stim_Processor(r'G:\Test_Data\2P\201211_L76_2P\1-010',
                        r'G:\Test_Data\2P\201211_L76_2P\1-010\Results\Stim_Frame_Align.pkl',
                        Sub_Dic_Generator('G8+90'),
                        cell_method = r'G:\Test_Data\2P\201211_L76_2P\1-010\Results\Global_Morpho\Global_Morpho.cell',
                        spike_train_path=r'G:\Test_Data\2P\201211_L76_2P\1-010\Results\dF_F_train.pkl'
                        )
Standard_Stim_Processor(r'G:\Test_Data\2P\201211_L76_2P\1-012',
                        r'G:\Test_Data\2P\201211_L76_2P\1-012\Results\Stim_Frame_Align.pkl',
                        Sub_Dic_Generator('OD_2P'),
                        cell_method = r'G:\Test_Data\2P\201211_L76_2P\1-012\Results\Global_Morpho\Global_Morpho.cell',
                        spike_train_path=r'G:\Test_Data\2P\201211_L76_2P\1-012\Results\dF_F_train.pkl'
                        )
Standard_Stim_Processor(r'G:\Test_Data\2P\201211_L76_2P\1-014',
                        r'G:\Test_Data\2P\201211_L76_2P\1-014\Results\Stim_Frame_Align.pkl',
                        Sub_Dic_Generator('RGLum4'),
                        cell_method = r'G:\Test_Data\2P\201211_L76_2P\1-014\Results\Global_Morpho\Global_Morpho.cell',
                        spike_train_path=r'G:\Test_Data\2P\201211_L76_2P\1-014\Results\dF_F_train.pkl'
                        )