# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 10:57:11 2021

@author: ZR

"""
from My_Wheels.Stim_Frame_Align import One_Key_Stim_Align
from My_Wheels.Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
from My_Wheels.Standard_Stim_Processor import One_Key_Frame_Graphs
#%% Stim Frame Align
One_Key_Stim_Align(r'K:\Test_Data\2P\210401_L76_2P\210401_L76_stimuli')
#%% Calculate each run subframes.
H7D4_para = Sub_Dic_Generator('Hue7Ori4',
                              para = {'Hue':['5R7_8','5Y7_8','5G7_8','10GY7_8','5B7_8','5P7_8','N7']})
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210401_L76_2P\1-012', H7D4_para,
                     alinged_sub_folder = r'\Results\Affined_Frames')
All_Green_para = Sub_Dic_Generator('Hue7Ori4',
                              para = {'Hue':['5G7_10','5G7_8','5G7_6','5G7_4','5G7_2','N7','5G5_6']})
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210401_L76_2P\1-013', All_Green_para,
                     alinged_sub_folder = r'\Results\Affined_Frames')
G16_para = Sub_Dic_Generator('G16_2P')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210401_L76_2P\1-011', G16_para,
                     alinged_sub_folder = r'\Results\Affined_Frames')
OD_para = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210401_L76_2P\1-009', OD_para,
                     alinged_sub_folder = r'\Results\Affined_Frames')
#%% Get cutted sub graph.
from My_Wheels.OI_Graph_Cutter import OI_Graph_Cutter
OI_Graph_Cutter(r'K:\Test_Data\L76_All_OI_Maps\201103_Maps', r'K:\Test_Data\2P\210401_L76_2P\_Location\Masks.png')
