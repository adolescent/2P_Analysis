# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 12:47:39 2021

@author: ZR
"""

from Standard_Aligner import Standard_Aligner
from Standard_Stim_Processor import One_Key_Frame_Graphs
from Stim_Frame_Align import One_Key_Stim_Align
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator

day_folder = r'F:\Test_Data\2P\210831_L76_2P'
Sa = Standard_Aligner(day_folder, [1,2,3,4,5,6,7],final_base='1-002')
Stim_Frame_Align = One_Key_Stim_Align(r'F:\Test_Data\2P\210831_L76_2P\210831_stimuli')
G16_Dic = Sub_Dic_Generator('G16_2P')
One_Key_Frame_Graphs(r'F:\Test_Data\2P\210831_L76_2P\1-002', G16_Dic)
OD_Dic = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'F:\Test_Data\2P\210831_L76_2P\1-006', OD_Dic)
Hue_Para = Sub_Dic_Generator('HueNOrien4',para = 'Default')
One_Key_Frame_Graphs(r'F:\Test_Data\2P\210831_L76_2P\1-007', Hue_Para)

from Cell_Find_From_Graph import Cell_Find_From_Mannual
cell_dic = Cell_Find_From_Mannual(r'F:\Test_Data\2P\210831_L76_2P\_Manual_Cell\Cell_Mask.png',
                                  average_graph_path = r'F:\Test_Data\2P\210831_L76_2P\_Manual_Cell\Global_Average.tif',boulder = 5)
from Standard_Cell_Generator import Standard_Cell_Generator
Scg = Standard_Cell_Generator('L76', '210831', r'F:\Test_Data\2P\210831_L76_2P', [1,2,3,4,5,6,7])
Scg.Generate_Cells()
day_folder = r'F:\Test_Data\2P\210831_L76_2P'

from Stimulus_Cell_Processor.Tuning_Property_Calculator import Tuning_Property_Calculator
all_tunings,tuning_checklist = Tuning_Property_Calculator(day_folder)
