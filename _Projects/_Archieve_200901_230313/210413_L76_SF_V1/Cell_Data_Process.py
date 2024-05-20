# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 10:34:54 2021

@author: ZR
"""

from Cell_Processor import Cell_Processor
from Standard_Parameters.Stim_Name_Tools import Stim_ID_Combiner
CP = Cell_Processor(r'K:\Test_Data\2P\210413_L76_2P')
G16_Para = Stim_ID_Combiner('G16_Dirs')
CP.Cell_Response_Maps('Run013', G16_Para,subshape = (3,8))
for i in range(CP.cell_num):
    c_cname = CP.all_cell_names[i]
    CP.Single_Cell_Plotter(c_cname,show_time = 0)