# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:12:31 2020

@author: ZR
"""

import My_Wheels.List_Operation_Kit as List_Tools
from My_Wheels.Translation_Align_Function import Translation_Alignment
#%% Initialization
data_path = [r'H:\Test_Data\2P\201201_L76_2P']
run_path = [
    '1-001',
    '1-002',
    '1-003',
    '1-004',
    '1-005',
    '1-006',
    '1-007',
    ]
all_run_names = List_Tools.List_Annex(data_path, run_path)
#%% Align
Translation_Alignment(all_run_names,base_mode=0)