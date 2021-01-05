# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 09:57:51 2021

@author: adolescent
"""
from My_Wheels.Translation_Align_Function import Translation_Alignment
import My_Wheels.List_Operation_Kit as List_Tools
import cv2
data_folder = [r'E:\Test_Data\2P\210101_L76_2P']
spon_run_folders = ['1-001','1-002']
all_spon_folders = List_Tools.List_Annex(data_folder, spon_run_folders)
Translation_Alignment(all_spon_folders,graph_shape=(376,352))
#%% Then align all other spon runs to Run01/Run02
after_spons = [r'E:\Test_Data\2P\210101_L76_2P\1-010',
               r'E:\Test_Data\2P\210101_L76_2P\1-018']
base_graph = cv2.imread(r'E:\Test_Data\2P\210101_L76_2P\1-002\Results\Global_Average_After_Align.tif',-1)
Translation_Alignment(after_spons,base_mode = 'input',input_base = base_graph,graph_shape = (376,352))
