# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 10:32:26 2021

@author: ZR
Step1, generate cell frames.
"""

import OS_Tools_Kit as ot
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
from Series_Analyzer.Cell_Activity_Evaluator import Spike_Count
import numpy as np

save_folder = r'D:\ZR\_My_Codes\2P_Analysis\_Projects\211027_Spon_Net_Correlation'
#%% Generate used cell frame here.
# Test 0831 data first
folder_831 = r'G:\Test_Data\2P\210831_L76_2P'
Run01_831_Frame = Pre_Processor(folder_831,'Run001')
counter,_ = Spike_Count(Run01_831_Frame)
counter = np.array(counter.mean(0))
Run03_831_Frame = Pre_Processor(folder_831,'Run003')
counter,_ = Spike_Count(Run03_831_Frame)
counter = np.array(counter.mean(0))
# Get used Frame
Run01_831_Frame = Pre_Processor(folder_831,'Run001',7200,99999)
cells_run01 = list(Run01_831_Frame.index)
cells_run03 = list(Run03_831_Frame.index)
common_cells = list(set(cells_run01)&set(cells_run03))
common_cells.sort()
selected_831_before = Run01_831_Frame.loc[common_cells]
selected_831_after = Run03_831_Frame.loc[common_cells]
ot.Save_Variable(save_folder, '0831_Before', selected_831_before)
ot.Save_Variable(save_folder, '0831_After', selected_831_after)
# Then 0920 data.
folder_920 = r'G:\Test_Data\2P\210920_L76_2P'
Run01_920_Frame = Pre_Processor(folder_920,'Run001')
counter,_ = Spike_Count(Run01_920_Frame)
counter = np.array(counter.mean(0))
Run03_920_Frame = Pre_Processor(folder_920,'Run003')
counter,_ = Spike_Count(Run03_920_Frame)
counter = np.array(counter.mean(0))
# All cells in run, so just plot them.
Run01_920_Frame = Pre_Processor(folder_920,'Run001',6000,99999)
ot.Save_Variable(save_folder, '0920_Before', Run01_920_Frame)
ot.Save_Variable(save_folder, '0920_After', Run03_920_Frame)




