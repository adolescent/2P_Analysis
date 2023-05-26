# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 22:02:51 2021

@author: ZR
"""

from My_Wheels.Standard_Aligner import Standard_Aligner

Sa = Standard_Aligner(r'K:\Test_Data\2P\210629_L76_2P', list(range(1,8)))
Sa.One_Key_Aligner()

from My_Wheels.Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'K:\Test_Data\2P\210629_L76_2P\210629_L76_2P_stimuli')


from My_Wheels.Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
from My_Wheels.Standard_Stim_Processor import One_Key_Frame_Graphs
G8_Para = Sub_Dic_Generator('G8+90')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210629_L76_2P\1-004',G8_Para)
OD_Para = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210629_L76_2P\1-006',OD_Para)
RG_Para = Sub_Dic_Generator('RGLum4')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210629_L76_2P\1-007',RG_Para)

from My_Wheels.Cell_Find_From_Graph import Cell_Find_From_Mannual
Cell_Find_From_Mannual(r'K:\Test_Data\2P\210629_L76_2P\_Manual_Cell\Cell_Mask.png',
                       average_graph_path=r'K:\Test_Data\2P\210629_L76_2P\_Manual_Cell\Global_Average.tif',boulder = 5)
from My_Wheels.Standard_Cell_Generator import Standard_Cell_Generator
Scg = Standard_Cell_Generator('L76', '210629', r'K:\Test_Data\2P\210629_L76_2P', list(range(1,8)))
Scg.Generate_Cells()
