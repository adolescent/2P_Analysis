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
