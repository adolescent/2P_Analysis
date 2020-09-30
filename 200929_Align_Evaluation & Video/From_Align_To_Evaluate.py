# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:38:22 2020

@author: ZR
"""

import My_Wheels.Graph_Operation_Kit as Graph_Tools
import My_Wheels.Calculation_Functions as Calculator
from My_Wheels.Translation_Align_Function import Translation_Alignment
#%% First part, Align unaligned runs.
data_folders = [
    r'D:\Test_Data\190412_L74_LM\1-001',
    r'D:\Test_Data\190412_L74_LM\1-002',
    r'D:\Test_Data\190412_L74_LM\1-003',
    r'D:\Test_Data\190412_L74_LM\1-004',
    ]
Translation_Alignment(data_folders,big_memory_mode = True)
#%% Then, choose the base graph and cut method, generate vector & distance change.
import My_Wheels.Graph_Cutter as 
base_graph = 