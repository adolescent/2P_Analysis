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
from Stimulus_Cell_Processor.Get_Average_Tuning import Get_Average_Tuning
import scipy.stats as stats
import numpy as np
from Mask_Tools.Two_Circle_Mask import Two_Circle_Mask
from Mask_Tools.Cell_In_Mask import Cell_In_Mask,Get_Cells_Mass_Center
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import OS_Tools_Kit as ot

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
# first, on raw data.
times  = 10000
corr_frame_raw = pd.DataFrame(columns = ['correlation','Tuning_difference'])

for i in tqdm(range(times)):
    c_mask_A,c_mask_B,_ = Two_Circle_Mask(radius = 70,dist = 200)
    cell_in_A,_ = Cell_In_Mask(all_cell_dic,acn,c_mask_A)
    cell_in_B,_ = Cell_In_Mask(all_cell_dic,acn,c_mask_B)
    cell_A_cen = Get_Cells_Mass_Center(all_cell_dic, cell_in_A)
    cell_B_cen = Get_Cells_Mass_Center(all_cell_dic, cell_in_B)
    dist = np.linalg.norm(cell_A_cen-cell_B_cen)
    cell_A_avr = Run01_Frame.loc[cell_in_A,:].mean(0)
    cell_B_avr = Run01_Frame.loc[cell_in_B,:].mean(0)
    corr,_ = stats.pearsonr(cell_A_avr,cell_B_avr)
    # get tuning difference here.
    Avr_Tuning_A = Get_Average_Tuning(cell_in_A, tuning_dic)
    Avr_Tuning_B = Get_Average_Tuning(cell_in_B, tuning_dic)
    c_tuning_diff = abs(Avr_Tuning_A-Avr_Tuning_B)
    corr_frame_raw.loc[i] = [corr,c_tuning_diff]
# Same calculation on PC1 regressed data.
times = 10000
corr_frame_regressed = pd.DataFrame(columns = ['correlation','Tuning_difference'])

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
    # get tuning difference here.
    Avr_Tuning_A = Get_Average_Tuning(cell_in_A, tuning_dic)
    Avr_Tuning_B = Get_Average_Tuning(cell_in_B, tuning_dic)
    c_tuning_diff = abs(Avr_Tuning_A-Avr_Tuning_B)
    corr_frame_regressed.loc[i] = [corr,c_tuning_diff]

ot.Save_Variable(r'G:\Test_Data\2P\210831_L76_2P\_All_Results', 'Corr_Dist_200_R_70_raw', corr_frame_raw)
ot.Save_Variable(r'G:\Test_Data\2P\210831_L76_2P\_All_Results', 'Corr_Dist_200_R_70_regressed', corr_frame_regressed)

#%% Then, we get cell by cell calculation here.
all_cell_dist = ot.Load_Variable(day_folder,'Cell_Distance.dist')
dist_corr_frame = pd.DataFrame(columns = ['Dist','Corr','PairName'])
dist_corr_frame_regressed = pd.DataFrame(columns = ['Dist','Corr','PairName'])
counter = 0
for i,cell_A in tqdm(enumerate(acn)):
    for j in range(i+1,len(acn)):
        cell_B = acn[j]
        c_r,_ = stats.pearsonr(Run01_Frame.loc[cell_A,:],Run01_Frame.loc[cell_B,:])
        c_r_reg,_ = stats.pearsonr(Run01_Frame_regressed.loc[cell_A,:],Run01_Frame.loc[cell_B,:])
        c_dist = all_cell_dist.loc[cell_A,cell_B]
        c_pairname = cell_A+'-'+cell_B
        dist_corr_frame.loc[counter,:] = [c_dist,c_r,c_pairname]
        dist_corr_frame_regressed.loc[counter,:] = [c_dist,c_r_reg,c_pairname]
        counter +=1

dist_corr_frame_regressed['Dist'] = dist_corr_frame_regressed['Dist'].astype('f8')
dist_corr_frame_regressed['Corr'] = dist_corr_frame_regressed['Corr'].astype('f8')
sns.lmplot(data = dist_corr_frame_regressed,x = 'Dist',y = 'Corr',scatter_kws={"s": 3})
dist_corr_frame['Dist'] = dist_corr_frame['Dist'].astype('f8')
dist_corr_frame['Corr'] = dist_corr_frame['Corr'].astype('f8')
sns.lmplot(data = dist_corr_frame,x = 'Dist',y = 'Corr',scatter_kws={"s": 3})

#%% Dist adjustable correlation
dists = list(range(270,350,10))
times = 3000
corr_frame_raw = pd.DataFrame(columns = ['Correlation','Tuning_Difference','Dist'])
corr_frame_regressed = pd.DataFrame(columns = ['Correlation','Tuning_Difference','Dist'])
counter = 0
for i,c_dist in enumerate(dists):
    for j in tqdm(range(times)):
        c_mask_A,c_mask_B,_ = Two_Circle_Mask(radius = 70,dist = c_dist)
        cell_in_A,_ = Cell_In_Mask(all_cell_dic,acn,c_mask_A)
        cell_in_B,_ = Cell_In_Mask(all_cell_dic,acn,c_mask_B)
        cell_A_avr = Run01_Frame.loc[cell_in_A,:].mean(0)
        cell_B_avr = Run01_Frame.loc[cell_in_B,:].mean(0)
        corr,_ = stats.pearsonr(cell_A_avr,cell_B_avr)
        Avr_Tuning_A = Get_Average_Tuning(cell_in_A, tuning_dic)
        Avr_Tuning_B = Get_Average_Tuning(cell_in_B, tuning_dic)
        c_tuning_diff = abs(Avr_Tuning_A-Avr_Tuning_B)
        corr_frame_raw.loc[counter] = [corr,c_tuning_diff,c_dist]
        cell_A_avr_reg = Run01_Frame_regressed.loc[cell_in_A,:].mean(0)
        cell_B_avr_reg = Run01_Frame_regressed.loc[cell_in_B,:].mean(0)
        corr_reg,_ = stats.pearsonr(cell_A_avr_reg,cell_B_avr_reg)
        corr_frame_regressed.loc[counter] = [corr_reg,c_tuning_diff,c_dist]
        counter += 1
ot.Save_Variable(r'G:\Test_Data\2P\210831_L76_2P\_All_Results','Corr_Dist_270_350_R_70_raw', corr_frame_raw)
ot.Save_Variable(r'G:\Test_Data\2P\210831_L76_2P\_All_Results','Corr_Dist_270_350_R_70_regressed', corr_frame_regressed)
