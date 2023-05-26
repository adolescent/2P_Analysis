# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 20:00:25 2022

@author: ZR
"""


from My_Wheels.Caiman_API.One_Key_Caiman import One_Key_Caiman
from Stim_Frame_Align import One_Key_Stim_Align
from My_Wheels.Caiman_API.Condition_Response_Generator import All_Cell_Condition_Generator
from My_Wheels.Caiman_API.Map_Generators_CAI import One_Key_T_Map
from My_Wheels.Caiman_API.Precess_Pipeline import Preprocess_Pipeline

#%%
day_folder = r'G:\Test_Data\2P\220630_L76_2P'
pp = Preprocess_Pipeline(day_folder, [1,3,6,7,8],orien_run = 'Run007',color_run = 'Run008')
pp.Do_Preprocess()
 