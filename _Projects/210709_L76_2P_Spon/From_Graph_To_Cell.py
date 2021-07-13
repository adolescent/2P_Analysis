# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 10:34:23 2021

@author: ZR
"""

from Standard_Aligner import Standard_Aligner
Sa = Standard_Aligner(r'K:\Test_Data\2P\210708_L76_2P', list(range(1,21)),final_base = '1-017')
Sa.One_Key_Aligner()

from My_Wheels.Standard_Stim_Processor import One_Key_Frame_Graphs
from My_Wheels.Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator

One_Key_Frame_Graphs(data_folder, sub_dic)