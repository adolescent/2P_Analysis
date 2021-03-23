# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 12:49:43 2021

@author: ZR
"""

import My_Wheels.List_Operation_Kit as lt
import My_Wheels.Graph_Operation_Kit as gt
import My_Wheels.OS_Tools_Kit as ot
from My_Wheels.Translation_Align_Function import Translation_Alignment
import cv2
from My_Wheels.Affine_Alignment import Affine_Aligner_Gaussian


day_folder = r'K:\Test_Data\2P\210320_L76_2P'
run_folders = lt.Run_Name_Producer_2P(list(range(1,16)))
all_run_folder = lt.List_Annex([day_folder], run_folders)
# Align spon first
Translation_Alignment(all_run_folder[0:2])
# Then Align all rest runs.
base_graph = cv2.imread(r'K:\Test_Data\2P\210320_L76_2P\1-001\Results\Global_Average_After_Align.tif',-1)
Translation_Alignment(all_run_folder[2:],base_mode = 'input',input_base = base_graph,align_range = 35)
# Find some small mismatch, try affine methods.
affine_base = cv2.imread(r'K:\Test_Data\2P\210320_L76_2P\_Affine_Affairs\Affine_Base.tif',-1)
Affine_Aligner_Gaussian(all_run_folder[0], affine_base,write_file = False)
for i in range(2,15):
    Affine_Aligner_Gaussian(all_run_folder[i], affine_base,write_file = False)
# Get all aligned_average
all_avr = gt.Global_Averagor(all_run_folder)
gt.Show_Graph(gt.Clip_And_Normalize(all_avr,5), 'Global_Average_After_Affine', r'K:\Test_Data\2P\210320_L76_2P\_Affine_Affairs')
# Then align all stim runs
from My_Wheels.Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'K:\Test_Data\2P\210320_L76_2P\210320_L76_2P_stimuli')
# calculate frame submaps.
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
from Standard_Stim_Processor import One_Key_Frame_Graphs
G16_Para = Sub_Dic_Generator('G16_2P')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210320_L76_2P\1-005', G16_Para,alinged_sub_folder=r'\Results\Affined_Frames')
OD_Para = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210320_L76_2P\1-010', OD_Para,alinged_sub_folder=r'\Results\Affined_Frames')
S3D8_Para = Sub_Dic_Generator('Shape3Dir8')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210320_L76_2P\1-013', S3D8_Para,alinged_sub_folder=r'\Results\Affined_Frames')

