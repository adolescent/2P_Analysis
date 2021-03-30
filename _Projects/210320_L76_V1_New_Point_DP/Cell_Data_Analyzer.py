# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 14:37:51 2021

@author: ZR
"""

from My_Wheels.Standard_Parameters.Stim_Name_Tools import Stim_ID_Combiner
from My_Wheels.Cell_Processor import Cell_Processor
import My_Wheels.OS_Tools_Kit as ot
import matplotlib.pyplot as plt


day_folder = r'K:\Test_Data\2P\210320_L76_2P'
save_folder = day_folder+r'\_All_Results'
ot.mkdir(save_folder)
#%% Analyze Run05-G16 First.
G16_CP = Cell_Processor(day_folder, 'Run005')
all_cell_name = G16_CP.all_cell_names
Ori_IDs = Stim_ID_Combiner('G16_Oriens')
sub_sf = save_folder+r'\G16_Oriens'
ot.mkdir(sub_sf)
for i in range(len(all_cell_name)):
    _,raw_data,_ = G16_CP.Single_Cell_Response_Data(Ori_IDs, all_cell_name[i])
    ot.Save_Variable(sub_sf, all_cell_name[i], raw_data,'.raw')
    test_fig = G16_CP.Average_Response_Map()
    test_fig.savefig(sub_sf+r'\\'+all_cell_name[i]+'_Response.png',dpi = 180)
    plt.clf()
#%% Then directions
Dir_IDs = Stim_ID_Combiner('G16_Dirs')
sub_sf = save_folder+r'\G16_Dirs'
ot.mkdir(sub_sf)
for i in range(len(all_cell_name)):
    _,raw_data,_ = G16_CP.Single_Cell_Response_Data(Dir_IDs, all_cell_name[i])
    ot.Save_Variable(sub_sf, all_cell_name[i], raw_data,'.raw')
    test_fig = G16_CP.Average_Response_Map()
    test_fig.savefig(sub_sf+r'\\'+all_cell_name[i]+'_Response.png',dpi = 180)
    plt.clf()
    
#%% Then Analyze Run15-Color7Dir8
C7D8_CP = Cell_Processor(day_folder, 'Run015')
C7D9_Colors = Stim_ID_Combiner('Color7Dir8_Colors')
sub_sf = save_folder+r'\C7D8_Colors'
ot.mkdir(sub_sf)
for i in range(len(all_cell_name)):
    _,raw_data,_ = C7D8_CP.Single_Cell_Response_Data(C7D9_Colors, all_cell_name[i])
    ot.Save_Variable(sub_sf, all_cell_name[i], raw_data,'.raw')
    test_fig = C7D8_CP.Average_Response_Map()
    test_fig.savefig(sub_sf+r'\\'+all_cell_name[i]+'_Response.png',dpi = 180)
    plt.clf()