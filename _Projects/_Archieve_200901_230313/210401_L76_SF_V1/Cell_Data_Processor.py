# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:13:17 2021

@author: ZR
"""
from Cell_Processor import Cell_Processor

CP = Cell_Processor(r'K:\Test_Data\2P\210401_L76_2P')
for i in range(CP.cell_num):
    c_name = CP.all_cell_names[i]
    CP.Single_Cell_Plotter(c_name,show_time = 0)