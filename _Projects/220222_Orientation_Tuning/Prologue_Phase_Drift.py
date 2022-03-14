# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:51:58 2022

@author: ZR
Get phase drifted data, trying to say it...
"""

import OS_Tools_Kit as ot
import cv2
import seaborn as sns
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
from Series_Analyzer.Cell_Frame_PCA import Do_PCA,PCA_Regression
import matplotlib.pyplot as plt
from Series_Analyzer.Single_Component_Visualize import Single_Mask_Visualize
from Stimulus_Cell_Processor.Get_Tuning import Get_Tuned_Cells
import scipy.stats as stats
import numpy as np
import pandas as pd
from Mask_Tools.Two_Circle_Mask import Two_Circle_Mask
from Mask_Tools.Cell_In_Mask import Cell_In_Mask,Get_Cells_Mass_Center
from tqdm import tqdm
from Stimulus_Cell_Processor.Get_Average_Tuning import Get_Average_Tuning

#%% read in 
day_folder = r'G:\Test_Data\2P\210831_L76_2P'
all_cell_dic = ot.Load_Variable(day_folder,'L76_210831A_All_Cells.ac')
Run01_Frame = Pre_Processor(day_folder,start_time = 7000)
acn = list(Run01_Frame.index)
Corr_Frames = pd.DataFrame(columns = ['CellA','CellB','origin_corr','intersected_forr','phase_shifted_corr'])
#%% Intersect Run01 Frame into 512 times
from scipy import interpolate
frame_num = Run01_Frame.shape[1]
inter_index = 512# how many times will series be.
Run01_Frame_intersected = pd.DataFrame()
for i,cc in tqdm(enumerate(acn)):
    c_train = Run01_Frame.loc[cc,:]
    origin_x = np.linspace(0,frame_num-1,frame_num)
    new_x = np.linspace(0,frame_num-1,frame_num*inter_index)
    f = interpolate.interp1d(origin_x,c_train,kind='linear')
    y_new = f(new_x)
    Run01_Frame_intersected[cc] = y_new
    
del f,c_train,origin_x,new_x,y_new,cc
Run01_Frame_intersected = Run01_Frame_intersected.T
#%% Shift data by phase....

Run01_Frame_intersected_shifted = pd.DataFrame()
for i,cc in enumerate(acn):
    cc_info = all_cell_dic[cc]['Cell_Info']
    y = int(cc_info.centroid[0])
    shift_dist = 512-y# shift by y.
    c_inter_series = np.array(Run01_Frame_intersected.loc[cc,:])
    cutted_inter_series = c_inter_series[shift_dist:]
    shifted_c_inter_series = np.pad(cutted_inter_series,(0,shift_dist))
    Run01_Frame_intersected_shifted[cc] = shifted_c_inter_series
# drop last frame.
Run01_Frame_intersected_shifted = Run01_Frame_intersected_shifted.iloc[:-512,:].T    
    
#%% Calculate correlation in different groups.
Correlation_Compare_Frame = pd.DataFrame(columns = ['CellA','CellB','Raw_Corr','Intersected_Corr','Shifted_Corr'])
counter = 0
for i,cellA in tqdm(enumerate(acn)):
    for j in range(i+1,len(acn)):
        cellB = acn[j]
        raw_corr,_ = stats.pearsonr(Run01_Frame.loc[cellA,:],Run01_Frame.loc[cellB,:])
        intersected_corr,_ = stats.pearsonr(Run01_Frame_intersected.loc[cellA,:],Run01_Frame_intersected.loc[cellB,:])
        shifted_corr,_ = stats.pearsonr(Run01_Frame_intersected_shifted.loc[cellA,:],Run01_Frame_intersected_shifted.loc[cellB,:])
        Correlation_Compare_Frame.loc[counter,:] = [cellA,cellB,raw_corr,intersected_corr,shifted_corr]
        counter += 1
        
ot.Save_Variable(r'G:\Test_Data\2P\210831_L76_2P\_All_Results','Corr_Intersection_Compare', Correlation_Compare_Frame)

#%% Compare different kind of correlations.
Correlation_Compare_Frame['Raw_vs_Inter'] = Correlation_Compare_Frame['Raw_Corr']-Correlation_Compare_Frame['Intersected_Corr']
Correlation_Compare_Frame['Inter_vs_Shifted'] = Correlation_Compare_Frame['Intersected_Corr']-Correlation_Compare_Frame['Shifted_Corr']
Correlation_Compare_Frame['Raw_vs_Shifted'] = Correlation_Compare_Frame['Raw_Corr']-Correlation_Compare_Frame['Shifted_Corr']

#%% Compare 
tuning_dic = ot.Load_Variable(r'G:\Test_Data\2P\210831_L76_2P\All_Tuning_Property.tuning')
times  = 3000
corr_frame_raw = pd.DataFrame(columns = ['correlation_raw','correlation_shifted','Tuning_difference',])

for i in tqdm(range(times)):
    c_mask_A,c_mask_B,_ = Two_Circle_Mask(radius = 70,dist = 200)
    cell_in_A,_ = Cell_In_Mask(all_cell_dic,acn,c_mask_A)
    cell_in_B,_ = Cell_In_Mask(all_cell_dic,acn,c_mask_B)
    cell_A_cen = Get_Cells_Mass_Center(all_cell_dic, cell_in_A)
    cell_B_cen = Get_Cells_Mass_Center(all_cell_dic, cell_in_B)
    #dist = np.linalg.norm(cell_A_cen-cell_B_cen)
    cell_A_avr = Run01_Frame.loc[cell_in_A,:].mean(0)
    cell_B_avr = Run01_Frame.loc[cell_in_B,:].mean(0)
    cell_A_shifted = Run01_Frame_intersected_shifted.loc[cell_in_A,:].mean(0)
    cell_B_shifted = Run01_Frame_intersected_shifted.loc[cell_in_B,:].mean(0)
    
    corr,_ = stats.pearsonr(cell_A_avr,cell_B_avr)
    corr_shifted,_ = stats.pearsonr(cell_A_shifted,cell_B_shifted)
    # get tuning difference here.
    Avr_Tuning_A = Get_Average_Tuning(cell_in_A, tuning_dic)
    Avr_Tuning_B = Get_Average_Tuning(cell_in_B, tuning_dic)
    c_tuning_diff = abs(Avr_Tuning_A-Avr_Tuning_B)
    corr_frame_raw.loc[i] = [corr,corr_shifted,c_tuning_diff]
    
ot.Save_Variable(r'G:\Test_Data\2P\210831_L76_2P','Mask_Corr_3000',corr_frame_raw)
# get compare graph.

compare_plot_graph = pd.DataFrame(columns = ['Corr','Tuning_Diff','Type'])
counter = 0
for i in tqdm(range(3000)):
    c_corr = corr_frame_raw.loc[i,'correlation_raw']
    c_corr_shifted = corr_frame_raw.loc[i,'correlation_shifted']
    c_td = corr_frame_raw.loc[i,'Tuning_difference']
    compare_plot_graph.loc[counter] = [c_corr,c_td,'Raw']
    compare_plot_graph.loc[counter+3000] = [c_corr_shifted,c_td,'Shifted']
    counter+=1

sns.lmplot(data = compare_plot_graph,x = 'Tuning_Diff',y = 'Corr',hue = 'Type',scatter_kws={'s':3})
#%% linear regression
from sklearn.linear_model import LinearRegression
X = (corr_frame_raw.loc[:,['Tuning_difference']])
Y = (corr_frame_raw.loc[:,['correlation_raw']])
model = LinearRegression()
model.fit(X,Y)


