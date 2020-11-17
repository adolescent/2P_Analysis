# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:33:19 2020

@author: adolescent
"""
from My_Wheels.Translation_Align_Function import Translation_Alignment
import My_Wheels.List_Operation_Kit as List_Tools
#%% Cell1 Align part. We use first 3 run and latter ones to match .Base graph is Run1-002 Before Average.
data_folder = r'E:\Test_Data\2P\201111_L76_LM'
run_folder =['1-001','1-002','1-003','1-009','1-012','1-013']
all_folders = List_Tools.List_Annex([data_folder], run_folder)
Translation_Alignment(all_folders,base_mode=1,align_range=35,align_boulder=35)
'''Attention here,1-012 and 1-013 have more movement than 20pix, making this hard to use.'''

