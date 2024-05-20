# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:51:24 2021

@author: ZR
"""
from Standard_Aligner import Standard_Aligner
from Standard_Stim_Processor import One_Key_Frame_Graphs
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
from Stim_Frame_Align import One_Key_Stim_Align



day_folder = r'F:\Test_Data\2P\210920_L76_2P'
Sa = Standard_Aligner(day_folder, [1,2,3,4,5,6,7])
Sa.One_Key_Aligner()
One_Key_Stim_Align(r'F:\Test_Data\2P\210920_L76_2P\210920_stimuli')

# Get only frame graph
G16_Para = Sub_Dic_Generator('G16_2P')
One_Key_Frame_Graphs(r'F:\Test_Data\2P\210920_L76_2P\1-002', G16_Para)
OD_Para = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'F:\Test_Data\2P\210920_L76_2P\1-006', OD_Para)
Hue_Para = Sub_Dic_Generator('HueNOrien4',para = 'Default')
One_Key_Frame_Graphs(r'F:\Test_Data\2P\210920_L76_2P\1-007', Hue_Para)

#%% Get cell data
from Cell_Find_From_Graph import Cell_Find_From_Mannual
from Standard_Cell_Generator import Standard_Cell_Generator
all_cell = Cell_Find_From_Mannual(r'F:\Test_Data\2P\210920_L76_2P\_Manual_Cell\Cell_Mask.png',
                                  r'F:\Test_Data\2P\210920_L76_2P\_Manual_Cell\Global_Average.tif',5)
Scg = Standard_Cell_Generator('L76','210920', day_folder,[1,2,3,4,5,6,7])
Scg.Generate_Cells()
from Stimulus_Cell_Processor.Tuning_Property_Calculator import Tuning_Property_Calculator
tuning_dic,tuning_checklist = Tuning_Property_Calculator(day_folder)

from Stimulus_Cell_Processor.T_Map_Generator import One_Key_T_Maps
G16_t_info = One_Key_T_Maps(day_folder,'Run002', 'G16_2P')
OD_t_info = One_Key_T_Maps(day_folder,'Run006', 'OD_2P')
Hue_t_info = One_Key_T_Maps(day_folder, 'Run007','HueNOrien4',para = 'Default')





