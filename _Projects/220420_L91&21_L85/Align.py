# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 20:56:36 2022

@author: ZR
"""

from My_Wheels.Standard_Aligner import Standard_Aligner

day_folder1 = r'G:\Test_Data\2P\220420_L91'
day_folder2 = r'G:\Test_Data\2P\220421_L85'
Sa_91  = Standard_Aligner(day_folder1, [1,2,3,4,5,6,7,8],final_base = '1-003')
Sa_85 = Standard_Aligner(day_folder2, [1,2,3,4,5,6,7,8,9],final_base = '1-003')
#%% Do aling seperately.
Sa_91.One_Key_Aligner_No_Affine()
Sa_85.One_Key_Aligner_No_Affine()
#%% Align stims
from Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'G:\Test_Data\2P\220420_L91\220420_stimuli')
One_Key_Stim_Align(r'G:\Test_Data\2P\220421_L85\220421_stimuli')
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
OD_Para = Sub_Dic_Generator('OD_2P')
from Standard_Stim_Processor import One_Key_Frame_Graphs
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220420_L91\1-006', OD_Para)
G16_Para = Sub_Dic_Generator('G16_2P')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220420_L91\1-007', G16_Para)
Hue_Para = Sub_Dic_Generator('HueNOrien4',para = 'Default')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220420_L91\1-008', Hue_Para)
# Do the same on L85 data.
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220421_L85\1-007', OD_Para)
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220421_L85\1-008', G16_Para)
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220421_L85\1-009', Hue_Para)