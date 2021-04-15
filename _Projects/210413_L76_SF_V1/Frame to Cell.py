# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 10:57:19 2021

@author: ZR
"""
import My_Wheels.List_Operation_Kit as lt
import cv2
from My_Wheels.Translation_Align_Function import Translation_Alignment
from Tremble_Evaluator import Least_Tremble_Average_Graph
import My_Wheels.Graph_Operation_Kit as gt
from My_Wheels.Affine_Alignment import Affine_Aligner_Gaussian
import My_Wheels.OS_Tools_Kit as ot

day_folder = [r'K:\Test_Data\2P\210413_L76_2P']
run_folders = lt.Run_Name_Producer_2P(list(range(1,15)))
all_runname = lt.List_Annex(day_folder, run_folders)
#%% First, align 
base = cv2.imread(r'K:\Test_Data\2P\210413_L76_2P\1-002\1_results\Averagepic-aftCor-3.tif',-1)
Translation_Alignment(all_runname,base_mode = 'input',input_base = base,align_range = 35)
#%% Second, stim maps.
from My_Wheels.Standard_Stim_Processor import One_Key_Frame_Graphs
from My_Wheels.Stim_Frame_Align import One_Key_Stim_Align
from My_Wheels.Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
One_Key_Stim_Align(r'K:\Test_Data\2P\210413_L76_2P\210413_L76_2P_stimuli')
OD_Para = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210413_L76_2P\1-007', OD_Para)
G16_Para = Sub_Dic_Generator('G16_2P')
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210413_L76_2P\1-013', G16_Para)
H11O4_Para = Sub_Dic_Generator('HueNOrien4',para = {'Hue':['Red0.6','Red0.5','Red0.4','Red0.3','Red0.2',
                                                           'Yellow','Green','Cyan','Blue','Purple','White'],
                                                    'SP':[('Red0.6','Red0.5'),('Red0.5','Red0.4'),('Red0.4','Red0.3'),('Red0.3','Red0.2'),
                                                          ('Red0.6','Green'),('Red0.5','Green'),('Red0.4','Green'),('Red0.3','Green'),('Red0.2','Green')]})
One_Key_Frame_Graphs(r'K:\Test_Data\2P\210413_L76_2P\1-014', H11O4_Para)

