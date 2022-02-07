# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 16:07:48 2022

@author: adolescent
"""
import OS_Tools_Kit as ot
import cv2
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
from Series_Analyzer.Cell_Frame_PCA import Do_PCA,PCA_Regression
import matplotlib.pyplot as plt
from Series_Analyzer.Single_Component_Visualize import Single_Mask_Visualize
from Stimulus_Cell_Processor.Get_Tuning import Get_Tuned_Cells
import scipy.stats as stats
import numpy as np
from Mask_Tools.Two_Circle_Mask import Two_Circle_Mask
from Mask_Tools.Cell_In_Mask import Cell_In_Mask,Get_Cells_Mass_Center

from tqdm import tqdm
#%% Read in mask and all cells.
day_folder = r'G:\Test_Data\2P\210831_L76_2P'
Run01_Frame = Pre_Processor(day_folder,start_time = 7000)
tuning_dic = ot.Load_Variable(day_folder,r'All_Tuning_Property.tuning')
comp,info,weight = Do_PCA(Run01_Frame)
Run01_Frame_regressed = PCA_Regression(comp, info, weight,var_ratio = 0.75)
all_cell_dic = ot.Load_Variable(day_folder,r'L76_210831A_All_Cells.ac')
del comp,info,weight
ot.Save_Variable(day_folder, 'PC1_Regressed_Run01', Run01_Frame_regressed)
acn = list(Run01_Frame_regressed.index)
#%% corr vs tuning similarity with same distance
times  = 3
corr_
for i in tqdm(range(times)):
    c_mask_A,c_mask_B,_ = Two_Circle_Mask(radius = 70,dist = 200)
    cell_in_A,_ = Cell_In_Mask(all_cell_dic,acn,c_mask_A)
    cell_in_B,_ = Cell_In_Mask(all_cell_dic,acn,c_mask_B)
    cell_A_cen = Get_Cells_Mass_Center(all_cell_dic, cell_in_A)
    cell_B_cen = Get_Cells_Mass_Center(all_cell_dic, cell_in_B)
    dist = np.linalg.norm(cell_A_cen-cell_B_cen)
    cell_A_avr = Run01_Frame_regressed.loc[cell_in_A,:].mean(0)
    cell_B_avr = Run01_Frame_regressed.loc[cell_in_B,:].mean(0)
    corr,_ = stats.pearsonr(cell_A_avr,cell_B_avr)
    
    
    
    