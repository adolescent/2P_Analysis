# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 09:40:06 2021

@author: adolescent
"""
from My_Wheels.Translation_Align_Function import Translation_Alignment
import My_Wheels.List_Operation_Kit as List_Tools
from My_Wheels.Stim_Frame_Align import One_Key_Stim_Align

day_folder = [r'G:\Test_Data\2P\210202_L76LM_2P']
run_folders = [
    '1-001',
    '1-003',
    '1-004',
    '1-005',
    '1-006',
    '1-007',
    '1-008',
    '1-009',
    '1-010',
    '1-011'
    ]
all_folders = List_Tools.List_Annex(day_folder, run_folders)
Translation_Alignment(all_folders)
One_Key_Stim_Align(r'G:\Test_Data\2P\210202_L76LM_2P\210202_L76_stimuli')
