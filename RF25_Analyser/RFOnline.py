# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 12:45:34 2020

@author: ZR

This program is used to run RF25 Online.
A support txt will be read in the same folder.

"""
import My_Wheels.OS_Tools_Kit as OS_Tools
from My_Wheels.Translation_Align_Function import Translation_Alignment
from My_Wheels.Stim_Frame_Align import Stim_Frame_Align
import My_Wheels.Cell_Find_From_Graph
#%% First, read in config file. 
# All Read in shall be in this part to avoid bugs = =
f = open('Config.punch','r')
config_info = f.readlines()
del f
frame_folder = config_info[3][:-1]# Remove '\n'
stim_folder = config_info[6][:-1]# Remove '\n'
cap_freq = float(config_info[9])
frame_thres = float(config_info[12])
#%% Second do graph align.
save_folder = frame_folder+r'\Results'
aligned_tif_folder = save_folder+r'\Aligned_Frames'
Translation_Alignment([frame_folder],big_memory_mode=True)
aligned_all_tif_name = OS_Tools.Get_File_Name(aligned_tif_folder)
#%% Third, Stim Frame Align
jmp_step = int(5000//cap_freq)
_,Frame_Stim_Dic = Stim_Frame_Align(stim_folder,frame_thres = frame_thres,jmp_step = jmp_step)
#%% Forth, generate On-Off graph and find cell.
Off_IDs = Frame_Stim_Dic[-1]
On_IDs = 