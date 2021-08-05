# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 16:19:58 2021

@author: ZR
"""

import OS_Tools_Kit as ot
from Filters import Signal_Filter
day_folder = r'K:\Test_Data\EEG\210804_L54_EEG'
all_file_name = ot.Get_File_Name(day_folder,'.smr')
Run01_Train = ot.Spike2_Reader(all_file_name[0],physical_channel=1)
Run02_Train = ot.Spike2_Reader(all_file_name[1],physical_channel=1)
raw_Run01_Train = Run01_Train['Channel_Data'].flatten()
raw_Run02_Train = Run02_Train['Channel_Data'].flatten()
Filted_Run01 = Signal_Filter(raw_Run01_Train,filter_para=(0.05,0.9))
Filted_Run02 = Signal_Filter(raw_Run02_Train,filter_para=(0.05,0.9))
