# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:23:43 2021

@author: ZR
"""

import My_Wheels.List_Operation_Kit as List_Tools
from My_Wheels.Translation_Align_Function import Translation_Alignment
import cv2

day_folder = [r'H:\Test_Data\2P\210112_L76_2P']
run_lists = [
    '1-001',
    '1-007',
    '1-008',
    '1-009',
    '1-011',
    '1-012',
    '1-013',
    '1-014',
    '1-015',
    '1-016'
    ]
all_run_folders = List_Tools.List_Annex(day_folder, run_lists)
#%% Align
Translation_Alignment([all_run_folders[0]])
base_graph = cv2.imread(r'H:\Test_Data\2P\210112_L76_2P\1-001\Results\Run_Average_After_Align.tif',-1)
Translation_Alignment(all_run_folders[1:],base_mode = 'input',input_base = base_graph)
