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
#%% Then calculate each graphs.
from My_Wheels.Cell_Find_From_Graph import Cell_Find_And_Plot
Cell_Find_And_Plot(r'J:\Test_Data\2P\210202_L76LM_2P\1-001\Results', r'Global_Average_After_Align.tif', 'All_Morpho',find_thres = 1.5)
cell_folder = r'J:\Test_Data\2P\210202_L76LM_2P\1-001\Results\All_Morpho'
from My_Wheels.Standard_Stim_Processor import One_Key_Stim_Maps
from My_Wheels.Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
C7D8_Para = Sub_Dic_Generator('Color7Dir8+90')# Caution! Direction checked. Stim program is a little different = =
One_Key_Stim_Maps(r'J:\Test_Data\2P\210202_L76LM_2P\1-008', cell_folder, C7D8_Para,have_blank = False)

SFTF_Para = Sub_Dic_Generator('SFTF',para = {'SF':[0.5,1,2],'TF':[2,4,8],'Dir':[270,315,0,45,90,135,180,225]})
One_Key_Stim_Maps(r'J:\Test_Data\2P\210202_L76LM_2P\1-013', cell_folder, SFTF_Para,have_blank = False)


