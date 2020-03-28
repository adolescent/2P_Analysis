# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:36:29 2020

@author: ZR

"""

import My_Wheels.Cross_Run_Align as Module_Align
import My_Wheels.List_Operation_Kit as List_Tools



#%% Step 1 Align
Day_Folder = [r'E:\Test_Data\200107_L80_LM']
Run_Folder = [
    '1-001',
    ]

Run_In_Align = List_Tools.List_Annex(Day_Folder, Run_Folder)
CRA = Module_Align.Cross_Run_Align(Run_In_Align)
CRA.Do_Align()
#%% Step 2 Stim_Frame_Align
from My_Wheels.Stim_Frame_Align import Stim_Frame_Align
stim_folder = r'E:\Test_Data\200107_L80_LM\200107_L80_2P_stimuli\Run01_2P_G8'
a,b = Stim_Frame_Align(stim_folder)