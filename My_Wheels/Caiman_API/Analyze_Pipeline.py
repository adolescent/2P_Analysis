# -*- coding: utf-8 -*-
"""
Created on Sat May 21 16:42:11 2022

@author: adolescent

This Script shows how to generate caiman data and basic map from raw data.
"""

from My_Wheels.Caiman_API.One_Key_Caiman import One_Key_Caiman
import OS_Tools_Kit as ot
from Stim_Frame_Align import One_Key_Stim_Align
from My_Wheels.Caiman_API.Condition_Response_Generator import All_Cell_Condition_Generator,Cell_Response_Map
from My_Wheels.Caiman_API.Map_Generators_CAI import One_Key_T_Map
#%% First, do caiman find cell and get cell data.
day_folder = r'F:\_Data_Temp\220420_L91'
run_lists = [1,2,3,6,7,8]
Okc = One_Key_Caiman(day_folder, run_lists,boulder = (20,20,20,20))
Okc.Do_Caiman()
#%% After generate graph, do stim frame align and get all condition data.
all_subfolders = ot.Get_Sub_Folders(day_folder)
for i,c_folder in enumerate(all_subfolders):
    if 'stimuli' in c_folder:
        stim_folder = c_folder
One_Key_Stim_Align(stim_folder)
#%% Then, get condition data and generate condition responses.
all_condition_response = All_Cell_Condition_Generator(day_folder)
# plot response graph if you need 
from Standard_Parameters.Stim_Name_Tools import Stim_ID_Combiner
G16_Condition_Dics = Stim_ID_Combiner('G16_Oriens')
c_run_map = Cell_Response_Map(day_folder, G16_Condition_Dics,runname = 'Run007')
#%% Plot t map
One_Key_T_Map(day_folder, 'Run007', 'G16_2P')
#%% Get all cell tuning dictionary.
