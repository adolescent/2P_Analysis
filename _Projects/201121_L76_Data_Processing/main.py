# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 12:48:10 2020

@author: ZR
Codes to process L76-201121 Data
"""

from My_Wheels.Translation_Align_Function import Translation_Alignment
import My_Wheels.List_Operation_Kit as List_Tools
#%% First location
data_folder=  [r'E:\Test_Data\2P\201121_L76_LM']
run_name = [
    '1-001',
    '1-002',
    '1-003',
    '1-004',
    '1-008',
    ]
import cv2
base = cv2.imread(r'E:\Test_Data\2P\201121_L76_LM\Loc1~8_Base.tif',-1)
loc1_folders = List_Tools.List_Annex(data_folder, run_name)
Translation_Alignment([loc1_folders[0]],base_mode = 'global',input_base = base,align_range = 20)
