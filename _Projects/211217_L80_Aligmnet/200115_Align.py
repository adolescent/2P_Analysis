# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 10:59:09 2021

@author: ZR
"""

from My_Wheels.Standard_Aligner import Standard_Aligner 
#%% Loc15
day_folder = r'G:\Test_Data\2P\200115_L80_LM'
Loc15_useful_runs = [1,5,6,7,8]
Sa_Loc15 = Standard_Aligner(day_folder, Loc15_useful_runs)
Sa_Loc15.One_Key_Aligner_No_Affine()

#%% Loc1
day_folder = r'G:\Test_Data\2P\200115_L80_LM_Loc1'
Loc1_runs = [9,10,11,12,13,14]
Sa_Loc1 = Standard_Aligner(day_folder, Loc1_runs,final_base = '1-013')
Sa_Loc1.One_Key_Aligner_No_Affine()
