# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:36:29 2020

@author: ZR

"""

import My_Wheels.Cross_Run_Align as Module_Align
import My_Wheels.List_Operation_Kit as List_Tools



#%% Step 1 Align
Day_Folder = [r'E:\ZR\Data_Temp\191215_L77_2P']
Run_Folder = [
    'Run01_V4_L11U_D210_GA_RFlocation_shape3_Sti2degStep2deg',
    'Run02_V4_L11U_D210_GA_RFsize',
    'Run03_V4_L11U_D210_GA_RFlocation_shape3_Sti2degStep2deg'
    ]

Run_In_Align = List_Tools.List_Annex(Day_Folder, Run_Folder)
CRA = Module_Align.Cross_Run_Align(Run_In_Align)
CRA.Do_Align()

#%%