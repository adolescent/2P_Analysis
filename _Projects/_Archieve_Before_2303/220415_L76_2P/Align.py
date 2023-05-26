# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 19:36:48 2022

@author: ZR
"""

from Standard_Aligner import Standard_Aligner
day_folder = r'G:\Test_Data\2P\220415_L76'
Sa = Standard_Aligner(day_folder, [1,2,3,4,5,6,7,8])
Sa.One_Key_Aligner_No_Affine()

#%% Align graphs.
from Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'G:\Test_Data\2P\220415_L76\220415_L76_stimuli')
from Standard_Stim_Processor import One_Key_Frame_Graphs
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
G16_Para = Sub_Dic_Generator('G16_2P')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220415_L76\1-007', G16_Para)
OD_Para = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220415_L76\1-006', OD_Para)
Hue_Para = Sub_Dic_Generator('HueNOrien4',para = 'Default')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220415_L76\1-008', Hue_Para)



