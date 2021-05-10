# -*- coding: utf-8 -*-
"""
Created on Wed May  5 10:15:48 2021

@author: ZR
"""

from My_Wheels.Standard_Aligner import Standard_Aligner
SA = Standard_Aligner(r'K:\Test_Data\2P\210504_L76_2P', list(range(1,17)),final_base='1-001')
SA.One_Key_Aligner()

from My_Wheels.Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'K:\Test_Data\2P\210504_L76_2P\210504_L76_stimuli')

from My_Wheels.Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
from My_Wheels.Standard_Stim_Processor import One_Key_Frame_Graphs
OD_Para = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210504_L76_2P\1-010', OD_Para)
G16_Para = Sub_Dic_Generator('G16_2P')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210504_L76_2P\1-013', G16_Para)
S3D8_Para = Sub_Dic_Generator('Shape3Dir8')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210504_L76_2P\1-015', S3D8_Para)
H7O4_Para = Sub_Dic_Generator('HueNOrien4',para = {'Hue':['Red','Yellow','Green','Cyan','Blue','Purple','White']})
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210504_L76_2P\1-016', H7O4_Para)
#%% Get cell file then.
from My_Wheels.Cell_Find_From_Graph import Cell_Find_From_Mannual
cell_dic = Cell_Find_From_Mannual(r'K:\Test_Data\2P\210504_L76_2P\_Manual_Cell\Cell_Mask.png',average_graph_path = r'K:\Test_Data\2P\210504_L76_2P\_Manual_Cell\Global_Average.tif',boulder = 5)
from My_Wheels.Standard_Cell_Generator import Standard_Cell_Generator
SCG = Standard_Cell_Generator('L76', '210504', r'K:\Test_Data\2P\210504_L76_2P', list(range(1,17)))
SCG.Generate_Cells()


# Till now, we get pure cell data here.