# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:29:26 2022

@author: ZR
"""

from Standard_Aligner import Standard_Aligner

day_folder = r'D:\ZR\_Temp_Data\220505_L85'
Sa = Standard_Aligner(day_folder, [1,2,3,4,5,6,7],final_base = '1-003')
Sa.One_Key_Aligner_No_Affine()

from Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'D:\ZR\_Temp_Data\220506_L76_2P\220506_stimuli')
One_Key_Stim_Align(r'D:\ZR\_Temp_Data\220505_L85\220505_L85_stimuli')
One_Key_Stim_Align(r'D:\ZR\_Temp_Data\220504_L91\220504_L91_stimuli')

#%% Get stim graphs.
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
from Standard_Stim_Processor import One_Key_Frame_Graphs
G16_Para = Sub_Dic_Generator('G16_2P')
OD_Para = Sub_Dic_Generator('OD_2P')
Hue_Para = Sub_Dic_Generator('HueNOrien4',para = 'Default')

One_Key_Frame_Graphs(r'D:\ZR\_Temp_Data\220504_L91\1-002', G16_Para)
One_Key_Frame_Graphs(r'D:\ZR\_Temp_Data\220504_L91\1-006', OD_Para)
One_Key_Frame_Graphs(r'D:\ZR\_Temp_Data\220504_L91\1-007', Hue_Para)


One_Key_Frame_Graphs(r'D:\ZR\_Temp_Data\220505_L85\1-002', G16_Para)
One_Key_Frame_Graphs(r'D:\ZR\_Temp_Data\220505_L85\1-006', OD_Para)
One_Key_Frame_Graphs(r'D:\ZR\_Temp_Data\220505_L85\1-007', Hue_Para)

One_Key_Frame_Graphs(r'D:\ZR\_Temp_Data\220506_L76_2P\1-002', G16_Para)
One_Key_Frame_Graphs(r'D:\ZR\_Temp_Data\220506_L76_2P\1-006', OD_Para)
One_Key_Frame_Graphs(r'D:\ZR\_Temp_Data\220506_L76_2P\1-007', Hue_Para)