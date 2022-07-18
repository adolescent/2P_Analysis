# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:29:52 2022

@author: ZR
"""


from Caiman_API.One_Key_Caiman import One_Key_Caiman
import OS_Tools_Kit as ot
import Graph_Operation_Kit as gt
from Stim_Frame_Align import One_Key_Stim_Align
from Caiman_API.Condition_Response_Generator import All_Cell_Condition_Generator
from Caiman_API.Map_Generators_CAI import One_Key_T_Map
from Stimulus_Cell_Processor.Get_Cell_Tuning_Cai import Tuning_Calculator
import warnings
from Decorators import Timer
from Caiman_API.Precess_Pipeline import Preprocess_Pipeline

#%% process L91 data into usable format.
day_folder_91 = r'D:\ZR\_Temp_Data\220420_L91'
pp = Preprocess_Pipeline(day_folder_91, [1,2,3,6,7,8],orien_run = 'Run007',color_run = 'Run008')

#%%
