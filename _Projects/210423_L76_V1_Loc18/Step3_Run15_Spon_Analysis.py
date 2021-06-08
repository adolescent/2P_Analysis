# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:08:26 2021

@author: ZR
"""
import OS_Tools_Kit as ot
import numpy as np
import pandas as pd


day_folder = r'K:\Test_Data\2P\210423_L76_2P'
all_cell_dic = ot.Load_Variable(day_folder+r'\L76_210423A_All_Cells.ac')
all_cell_name = list(all_cell_dic.keys())
all_spon_series = 
for i in range(len(all_cell_name)):
    tc = all_cel_dic[all_cell_name[i]]
    if tc['In_Run']['Run015'] == True:
        