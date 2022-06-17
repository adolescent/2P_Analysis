# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 10:25:04 2022

@author: ZR
"""

from My_Wheels.Caiman_API.One_Key_Caiman import One_Key_Caiman

day_folder85 = r'D:\ZR\_Temp_Data\220608_L85_2P'
Okc_85 = One_Key_Caiman(day_folder85, [1,2,3,4,7,8,9],align_base = '1-004')
Okc_85.Do_Caiman()
# then get CR trains.
from Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'D:\ZR\_Temp_Data\220608_L85_2P\220608_L85_stimuli')
from My_Wheels.Caiman_API.Condition_Response_Generator import All_Cell_Condition_Generator
cell_condition_dic = All_Cell_Condition_Generator(day_folder85)
# Last plot T maps.
from My_Wheels.Caiman_API.Map_Generators_CAI import One_Key_T_Map
One_Key_T_Map(day_folder85, 'Run007', 'OD_2P')
One_Key_T_Map(day_folder85, 'Run008', 'G16_2P')
One_Key_T_Map(day_folder85, 'Run009', 'HueNOrien4')
#%% And pca here.
from Series_Analyzer.Cell_Frame_PCA_Cai import One_Key_PCA
day_folder85 = r'D:\ZR\_Temp_Data\220608_L85_2P'
comp,info,weight = One_Key_PCA(day_folder85, 'Run002',tag = 'Spon_Before',start_time = 4000)
comp,info,weight = One_Key_PCA(day_folder85, 'Run004',tag = 'Spon_After',start_time = 0)


