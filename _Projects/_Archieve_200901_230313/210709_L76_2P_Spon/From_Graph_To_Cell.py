# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 10:34:23 2021

@author: ZR
"""

from Standard_Aligner import Standard_Aligner
Sa = Standard_Aligner(r'K:\Test_Data\2P\210708_L76_2P', list(range(1,21)),final_base = '1-017')
Sa.One_Key_Aligner()
from My_Wheels.Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'K:\Test_Data\2P\210708_L76_2P\210709_L76_2P_stimuli')
from My_Wheels.Standard_Stim_Processor import One_Key_Frame_Graphs
from My_Wheels.Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
OD_Para = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210708_L76_2P\1-015', OD_Para)
G8_Para = Sub_Dic_Generator('G8+90')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210708_L76_2P\1-018', G8_Para)
RG_Para = Sub_Dic_Generator('RGLum4')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210708_L76_2P\1-020', RG_Para)

import My_Wheels.OS_Tools_Kit as ot
all_cell_dic = ot.Load_Variable(r'K:\Test_Data\2P\210629_L76_2P\L76_210629A_All_Cells.ac')
