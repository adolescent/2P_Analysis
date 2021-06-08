# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:58:10 2021

@author: ZR
"""

from My_Wheels.Standard_Aligner import Standard_Aligner
day_folder = r'K:\Test_Data\2P\210525_L76_2P'
SA = Standard_Aligner(day_folder,list(range(1,15)),final_base = '1-008')
SA.One_Key_Aligner()
from My_Wheels.Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'K:\Test_Data\2P\210525_L76_2P\210525_L76_stimuli')

from My_Wheels.Standard_Stim_Processor import One_Key_Frame_Graphs
from My_Wheels.Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
G8_Dic = Sub_Dic_Generator('G8+90')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210525_L76_2P\1-008', G8_Dic)
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210525_L76_2P\1-012', G8_Dic)
OD_Dic = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210525_L76_2P\1-009', OD_Dic)
H7O4_Dic = Sub_Dic_Generator('HueNOrien4',{'Hue':['Red','Yellow','Green','Cyan','Blue','Purple','White']})
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210525_L76_2P\1-014', H7O4_Dic)

from My_Wheels.Cell_Find_From_Graph import Cell_Find_From_Mannual
cells = Cell_Find_From_Mannual(r'K:\Test_Data\2P\210525_L76_2P\_Manual_Cell\Cell_Mask.png',average_graph_path=r'K:\Test_Data\2P\210525_L76_2P\_Manual_Cell\Global_Average.tif')
