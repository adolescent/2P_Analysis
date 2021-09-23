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
