# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:07:48 2021

@author: adolescent
"""
import My_Wheels.List_Operation_Kit as lt
import My_Wheels.OS_Tools_Kit as ot
from My_Wheels.Translation_Align_Function import Translation_Alignment
from My_Wheels.Affine_Alignment import Affine_Aligner_Gaussian
import cv2
import numpy as np
#%% Step1 Align

day_folder = r'E:\Test_Data\2P\210401_L76_2P'
run_lists = lt.Run_Name_Producer_2P(list(range(1,14)))
all_run_folder = lt.List_Annex([day_folder], run_lists)
# Align spon series 
Translation_Alignment([all_run_folder[0]],graph_shape = (324,300))
Translation_Alignment([all_run_folder[9]],graph_shape = (324,300))
# And Align all stim frames to get base graphs.
all_stim_folder = np.append(all_run_folder[1:9],all_run_folder[10:13])
Translation_Alignment(all_stim_folder)
# Then we use affine method.
affine_base = cv2.imread(r'E:\Test_Data\2P\210401_L76_2P\_Affine_Affairs\Affine_Base.tif',-1)
affine_base2 = cv2.imread(r'E:\Test_Data\2P\210401_L76_2P\_Affine_Affairs\Global_Average_After_Align.tif',-1)

Affine_Aligner_Gaussian(all_run_folder[1], affine_base2)
Affine_Aligner_Gaussian(all_run_folder[2], affine_base2)
Affine_Aligner_Gaussian(all_run_folder[3], affine_base2,good_match_prop = 0.15)
Affine_Aligner_Gaussian(all_run_folder[4], affine_base2,good_match_prop = 0.15)
Affine_Aligner_Gaussian(all_run_folder[5], affine_base2)
Affine_Aligner_Gaussian(all_run_folder[6], affine_base2)
Affine_Aligner_Gaussian(all_run_folder[7], affine_base2)
Affine_Aligner_Gaussian(all_run_folder[8], affine_base2)
Affine_Aligner_Gaussian(all_run_folder[10], affine_base2)
Affine_Aligner_Gaussian(all_run_folder[11], affine_base2)
Affine_Aligner_Gaussian(all_run_folder[12], affine_base2)

