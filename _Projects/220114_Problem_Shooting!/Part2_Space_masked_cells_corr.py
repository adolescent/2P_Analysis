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
corr_frame_raw_150_260 = ot.Load_Variable(r'G:\Test_Data\2P\210831_L76_2P\_All_Results\Corr_Dist_150_260_R_70_raw.pkl')
corr_frame_regressed_150_260 = ot.Load_Variable(r'G:\Test_Data\2P\210831_L76_2P\_All_Results\Corr_Dist_150_260_R_70_regressed.pkl')
corr_frame_raw_whole = pd.concat([corr_frame_raw_150_260,corr_frame_raw_270_340])
corr_frame_regressed_whole = pd.concat([corr_frame_regressed_150_260,corr_frame_regressed_270_340])
#%% statastic of all corr in 
raw_corr_by_dist = list(corr_frame_raw_whole.groupby(['Dist']))
step = 0.02
Dist_Tuning_Corr_Frame = pd.DataFrame(columns = ['Dist','Tuning_Group','Avr_Corr'])

counter = 0
for i in range(20):
    c_group = raw_corr_by_dist[i][1]
    c_dist = raw_corr_by_dist[i][0]
    c_group['Tuning_Group'] = c_group['Tuning_Difference']//step+1
    cg_tuning_lists = list(c_group.groupby(['Tuning_Group']))
    for j,c_tuning_group in enumerate(cg_tuning_lists):
        c_tuning_group_frame = c_tuning_group[1]
        c_tg = c_tuning_group[0]
        avr_corr = c_tuning_group_frame['Correlation'].mean()
        Dist_Tuning_Corr_Frame.loc[counter] = [c_dist,c_tg,avr_corr]
        counter +=1
     
Dist_Tuning_Corr_Frame['Group_Tuning_Max'] = Dist_Tuning_Corr_Frame['Tuning_Group']*step
ot.Save_Variable(r'G:\Test_Data\2P\210831_L76_2P\_All_Results', 'Corr_Dist_Tuning_Raw', Dist_Tuning_Corr_Frame)
# Do the same on regressed data.
regressed_corr_by_dist = list(corr_frame_regressed_whole.groupby(['Dist']))
step = 0.02
Dist_Tuning_Corr_Frame_Reg = pd.DataFrame(columns = ['Dist','Tuning_Group','Avr_Corr'])
counter = 0
for i in range(20):
    c_group = regressed_corr_by_dist[i][1]
    c_dist = regressed_corr_by_dist[i][0]
    c_group['Tuning_Group'] = c_group['Tuning_Difference']//step+1
    cg_tuning_lists = list(c_group.groupby(['Tuning_Group']))
    for j,c_tuning_group in enumerate(cg_tuning_lists):
        c_tuning_group_frame = c_tuning_group[1]
        c_tg = c_tuning_group[0]
        avr_corr = c_tuning_group_frame['Correlation'].mean()
        Dist_Tuning_Corr_Frame_Reg.loc[counter] = [c_dist,c_tg,avr_corr]
        counter +=1

Dist_Tuning_Corr_Frame_Reg['Group_Tuning_Max'] = Dist_Tuning_Corr_Frame['Tuning_Group']*step
ot.Save_Variable(r'G:\Test_Data\2P\210831_L76_2P\_All_Results', 'Corr_Dist_Tuning_Regressed', Dist_Tuning_Corr_Frame_Reg)

# Plot Dist-Tuning Similarity correlation heat map.
Dist_vs_Tuning_Diff_raw = Dist_Tuning_Corr_Frame.pivot(index = 'Dist',columns = 'Group_Tuning_Max',values = 'Avr_Corr')
Dist_vs_Tuning_Diff_regressed = Dist_Tuning_Corr_Frame_Reg.pivot(index = 'Dist',columns = 'Group_Tuning_Max',values = 'Avr_Corr')
sns.heatmap(Dist_vs_Tuning_Diff_raw,center = 0.55,vmax = 0.85)
sns.heatmap(Dist_vs_Tuning_Diff_regressed,center = 0)

#%% Get Cell by Cell results here.
cell_by_cell_tuning_dist = pd.DataFrame(columns = ['Cell_A','Cell_B','Dist','Corr','Tuning_Diff'])
counter = 0
for i,A_cell in tqdm(enumerate(acn)):
    for j in range(i+1,len(acn)):
        B_cell = acn[j]
        c_dist = all_cell_dist.loc[A_cell,B_cell]
        c_A_series = Run01_Frame.loc[A_cell,:]
        c_B_series = Run01_Frame.loc[B_cell,:]
        c_corr,_ = stats.pearsonr(c_A_series,c_B_series)
        c_A_tuning = tuning_dic[A_cell]['LE']['Cohen_D']
        c_B_tuning = tuning_dic[B_cell]['LE']['Cohen_D']
        c_tuning_diff = abs(c_A_tuning-c_B_tuning)
        cell_by_cell_tuning_dist.loc[counter] = [A_cell,B_cell,c_dist,c_corr,c_tuning_diff]
        counter +=1
# Do the same on regressed data.
cell_by_cell_tuning_dist_regressed = pd.DataFrame(columns = ['Cell_A','Cell_B','Dist','Corr','Tuning_Diff'])
counter = 0
for i,A_cell in tqdm(enumerate(acn)):
    for j in range(i+1,len(acn)):
        B_cell = acn[j]
        c_dist = all_cell_dist.loc[A_cell,B_cell]
        c_A_series = Run01_Frame_regressed.loc[A_cell,:]
        c_B_series = Run01_Frame_regressed.loc[B_cell,:]
        c_corr,_ = stats.pearsonr(c_A_series,c_B_series)
        c_A_tuning = tuning_dic[A_cell]['LE']['Cohen_D']
        c_B_tuning = tuning_dic[B_cell]['LE']['Cohen_D']
        c_tuning_diff = abs(c_A_tuning-c_B_tuning)
        cell_by_cell_tuning_dist_regressed.loc[counter] = [A_cell,B_cell,c_dist,c_corr,c_tuning_diff]
        counter +=1        
        
#%% Plot Series dist-tuning relationship.
dist_step = 20 # group step of distance.
tuning_step = 0.02 # step of tuning difference.
Dist_Tuning_Corr_Cell_Raw = pd.DataFrame(columns = ['Dist_Group','Tuning_Group','Avr_Corr'])
cell_by_cell_tuning_dist['Dist_group'] = cell_by_cell_tuning_dist['Dist']//dist_step+1
cell_by_cell_tuning_dist['Td_group'] = cell_by_cell_tuning_dist['Tuning_Diff']//tuning_step+1
counter = 0
tuning_groups = list(cell_by_cell_tuning_dist.groupby('Td_group'))
for i,c_tuning_group in enumerate(tuning_groups):
    c_tuning = c_tuning_group[0]
    c_tuning_group_frame = c_tuning_group[1]
    c_dist_groups = list(c_tuning_group_frame.groupby('Dist_group'))
    for j in range(len(c_dist_groups)):
        c_dist = c_dist_groups[j][0]
        c_tuning_avr = c_dist_groups[j][1]['Corr'].mean()
        Dist_Tuning_Corr_Cell_Raw.loc[counter] = [c_dist,c_tuning,c_tuning_avr]
        counter +=1
Dist_Tuning_Corr_Cell_Raw['Dist'] =  Dist_Tuning_Corr_Cell_Raw['Dist_Group']*dist_step
Dist_Tuning_Corr_Cell_Raw['Tuning_Diff'] =  round(Dist_Tuning_Corr_Cell_Raw['Tuning_Group']*tuning_step,3)
# Do the same on regressed data
Dist_Tuning_Corr_Cell_Reg = pd.DataFrame(columns = ['Dist_Group','Tuning_Group','Avr_Corr'])
cell_by_cell_tuning_dist_regressed['Dist_group'] = cell_by_cell_tuning_dist_regressed['Dist']//dist_step+1
cell_by_cell_tuning_dist_regressed['Td_group'] = cell_by_cell_tuning_dist_regressed['Tuning_Diff']//tuning_step+1
counter = 0
tuning_groups = list(cell_by_cell_tuning_dist_regressed.groupby('Td_group'))
for i,c_tuning_group in enumerate(tuning_groups):
    c_tuning = c_tuning_group[0]
    c_tuning_group_frame = c_tuning_group[1]
    c_dist_groups = list(c_tuning_group_frame.groupby('Dist_group'))
    for j in range(len(c_dist_groups)):
        c_dist = c_dist_groups[j][0]
        c_tuning_avr = c_dist_groups[j][1]['Corr'].mean()
        Dist_Tuning_Corr_Cell_Reg.loc[counter] = [c_dist,c_tuning,c_tuning_avr]
        counter +=1
Dist_Tuning_Corr_Cell_Reg['Dist'] =  Dist_Tuning_Corr_Cell_Reg['Dist_Group']*dist_step
Dist_Tuning_Corr_Cell_Reg['Tuning_Diff'] =  round(Dist_Tuning_Corr_Cell_Reg['Tuning_Group']*tuning_step,3)
ot.Save_Variable(r'G:\Test_Data\2P\210831_L76_2P\_All_Results', 'Corr_Dist_Tuning_Cell_Raw',Dist_Tuning_Corr_Cell_Raw)
ot.Save_Variable(r'G:\Test_Data\2P\210831_L76_2P\_All_Results', 'Corr_Dist_Tuning_Cell_Regressed',Dist_Tuning_Corr_Cell_Reg)
# get cell by cell heat map.
Corr_Heatmap_Cell = Dist_Tuning_Corr_Cell_Raw.pivot(index = 'Dist',columns = 'Tuning_Diff',values = 'Avr_Corr')
Corr_Heatmap_Cell_reg = Dist_Tuning_Corr_Cell_Reg.pivot(index = 'Dist',columns = 'Tuning_Diff',values = 'Avr_Corr')

#%% Do linear regression for cell by cell data.
from sklearn.linear_model import LinearRegression
X = cell_by_cell_tuning_dist_regressed.loc[:,['Dist','Tuning_Diff']]
Y = cell_by_cell_tuning_dist_regressed.loc[:,['Corr']]
model = LinearRegression()
model.fit(X,Y)
model.score(X,Y)
sns.pairplot(cell_by_cell_tuning_dist_regressed,x_vars = ['Dist','Tuning_Diff'] ,y_vars = 'Corr',plot_kws=dict(scatter_kws=dict(s=1)),aspect = 1.5,kind = 'reg',size = 7)
