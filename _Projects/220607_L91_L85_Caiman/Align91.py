# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 20:51:06 2022

@author: ZR
"""

from My_Wheels.Caiman_API.One_Key_Caiman import One_Key_Caiman
from Stim_Frame_Align import One_Key_Stim_Align
from My_Wheels.Caiman_API.Condition_Response_Generator import All_Cell_Condition_Generator
from My_Wheels.Caiman_API.Map_Generators_CAI import One_Key_T_Map



#%% Do the same thing on L91 data.
day_folder91 = r'D:\ZR\_Temp_Data\220609_L91_2P'
Okc_91 = One_Key_Caiman(day_folder91, [1,2,3,6,7,8],align_base = '1-003')
Okc_91.Do_Caiman()
# then get CR trains.
One_Key_Stim_Align(r'D:\ZR\_Temp_Data\220609_L91_2P\220609_L91_stimuli')
cell_condition_dic = All_Cell_Condition_Generator(day_folder91)
# Last plot T maps.
One_Key_T_Map(day_folder91, 'Run006', 'OD_2P')
One_Key_T_Map(day_folder91, 'Run007', 'G16_2P')
One_Key_T_Map(day_folder91, 'Run008', 'HueNOrien4')
#%% Do PCA here.
from Series_Analyzer.Cell_Frame_PCA_Cai import One_Key_PCA
day_folder91 = r'D:\ZR\_Temp_Data\220609_L91_2P'
comp,info,weight = One_Key_PCA(day_folder91, '1-001',tag = 'Spon_Before',start_time = 2000)
comp,info,weight = One_Key_PCA(day_folder91, '1-003',tag = 'Spon_After',start_time = 0)



