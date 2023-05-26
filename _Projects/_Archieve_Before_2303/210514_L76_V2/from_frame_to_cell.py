# -*- coding: utf-8 -*-
"""
Created on Fri May 14 23:29:07 2021

@author: ZR
"""
import My_Wheels.OS_Tools_Kit as ot
from My_Wheels.Standard_Aligner import Standard_Aligner
SA = Standard_Aligner(r'K:\Test_Data\2P\210514_L76_2P', list(range(1,18)),final_base = '1-016',trans_range=20)
SA.One_Key_Aligner()
from My_Wheels.Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'K:\Test_Data\2P\210514_L76_2P\210514_stimuli')

from My_Wheels.Standard_Stim_Processor import One_Key_Frame_Graphs
from My_Wheels.Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator

G16_Dic = Sub_Dic_Generator('G16_2P')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210514_L76_2P\1-016', G16_Dic)
OD_Dic = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210514_L76_2P\1-014', OD_Dic)
S3D8_Dic = Sub_Dic_Generator('Shape3Dir8')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210514_L76_2P\1-013', S3D8_Dic)

H7O4_Dic = Sub_Dic_Generator('HueNOrien4',{'Hue':['Red','Yellow','Green','Cyan','Blue','Purple','White']})
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210514_L76_2P\1-017', H7O4_Dic)

from My_Wheels.Cell_Find_From_Graph import Cell_Find_From_Mannual
Cell_Dic = Cell_Find_From_Mannual(r'K:\Test_Data\2P\210514_L76_2P\_Manual_Cell\Cell_Mask.png',r'K:\Test_Data\2P\210514_L76_2P\_Manual_Cell\Global_Average.tif',5)
from My_Wheels.Standard_Cell_Generator import Standard_Cell_Generator
SCG = Standard_Cell_Generator('L76', '210514', r'K:\Test_Data\2P\210514_L76_2P', list(range(1,18)))
SCG.Generate_Cells()
