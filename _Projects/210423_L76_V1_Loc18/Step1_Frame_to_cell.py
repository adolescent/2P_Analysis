# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 23:59:59 2021

@author: ZR
"""
from My_Wheels.Standard_Aligner import Standard_Aligner

day_folder = r'K:\Test_Data\2P\210423_L76_2P'

SA = Standard_Aligner(day_folder, list(range(1,17)),final_base = '1-001')
SA.One_Key_Aligner()
#%% Then Stim_Frame_Align
from My_Wheels.Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'K:\Test_Data\2P\210423_L76_2P\210423_L76_2P_stimuli')

#%% Then get basic stim maps.
from My_Wheels.Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
from My_Wheels.Standard_Stim_Processor import One_Key_Frame_Graphs
OD_Para = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210423_L76_2P\1-009', OD_Para)
G16_Para = Sub_Dic_Generator('G16_2P')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210423_L76_2P\1-014', G16_Para)
Hue_Para = Sub_Dic_Generator('HueNOrien4',para = {'Hue':['Red0.6',
                                                         'Red0.5',
                                                         'Red0.4',
                                                         'Red0.3',
                                                         'Red0.2',
                                                         'Yellow',
                                                         'Green',
                                                         'Cyan',
                                                         'Blue',
                                                         'Purple',
                                                         'White'
                                                         ]})
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210423_L76_2P\1-016', Hue_Para)
#%% Then Manual Cells
from My_Wheels.Cell_Find_From_Graph import Cell_Find_From_Mannual
celldic = Cell_Find_From_Mannual(r'K:\Test_Data\2P\210423_L76_2P\_Manual_Cell\Cell_Mask.png',
                                 average_graph_path=r'K:\Test_Data\2P\210423_L76_2P\_Manual_Cell\Global_Average.tif',boulder = 8)
#%% Then Cell Generator
from My_Wheels.Standard_Cell_Generator import Standard_Cell_Generator
SCG = Standard_Cell_Generator('L76', '210423', r'K:\Test_Data\2P\210423_L76_2P', runid_lists = list(range(1,17)))
SCG.Generate_Cells()
