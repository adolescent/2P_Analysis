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



