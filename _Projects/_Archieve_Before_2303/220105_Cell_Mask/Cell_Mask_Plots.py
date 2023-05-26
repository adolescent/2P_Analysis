# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:05:21 2022

@author: ZR
"""
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
import matplotlib.pyplot as plt
import OS_Tools_Kit as ot
from Series_Analyzer.Single_Component_Visualize import Single_Comp_Visualize
import pandas as pd
import scipy.stats as stats
from Series_Analyzer.Cell_Frame_PCA import Do_PCA,Compoment_Visualize,PCA_Regression
import random
import List_Operation_Kit as lt
import numpy as np
import seaborn as sns
import time
from Analyzer.Statistic_Tools import T_Test_Pair
#%%  Calculate dF/F train first
day_folder = r'G:\Test_Data\2P\210831_L76_2P'
Run01_0831_dF_F = Pre_Processor(day_folder,start_time = 7000,
                                passed_band = (0.005,0.3),base_mode = 'most_unactive')
all_cell_dic = ot.Load_Variable(day_folder,'L76_210831A_All_Cells.ac')
tunings = ot.Load_Variable(day_folder,'All_Tuning_Property.tuning')
acn = list(Run01_0831_dF_F.index)
# Get LE cells & RE cells
thres = 0.05
LE_cells = []
RE_cells = []
for i,cc in enumerate(acn):
    c_tuning_OD = tunings[cc]['LE']
    if (c_tuning_OD['t_value']>0 and c_tuning_OD['p_value']<thres):
        LE_cells.append(cc)
    elif (c_tuning_OD['t_value']<0 and c_tuning_OD['p_value']<thres):
        RE_cells.append(cc)
        
LE_mask = pd.Series(1,index = LE_cells)
RE_mask = pd.Series(1,index = RE_cells)
# Then, get LE/RE activation plots
LE_cell_trains = Run01_0831_dF_F.loc[LE_cells]
LE_train_avr = LE_cell_trains.mean(0)
RE_cell_trains = Run01_0831_dF_F.loc[RE_cells]
RE_train_avr = RE_cell_trains.mean(0)
all_cell_mean = Run01_0831_dF_F.loc[acn].mean(0)

plt.plot(all_cell_mean)
plt.plot(LE_train_avr)
plt.plot(RE_train_avr)
#PCA and visualize
comp,info,weights = Do_PCA(Run01_0831_dF_F)
_ = Compoment_Visualize(comp, all_cell_dic, r'C:\Users\ZR\Desktop\temp')
#%% Calculate centered train here.
Run01_0831_centered = Pre_Processor(day_folder,start_time = 7000,
                                    passed_band = (0.005,0.3),base_mode = 'average')

LE_cell_trains_cen = Run01_0831_centered.loc[LE_cells]
LE_train_avr_cen = LE_cell_trains_cen.mean(0)
RE_cell_trains_cen = Run01_0831_centered.loc[RE_cells]
RE_train_avr_cen = RE_cell_trains_cen.mean(0)
rest_cells = lt.List_Subtraction(acn, LE_cells+RE_cells)
rest_cell_trains_cen = Run01_0831_centered.loc[rest_cells]
all_cell_mean_cen = Run01_0831_centered.loc[acn].mean(0)

plt.plot(rest_cell_mean_cen)
#plt.plot(all_cell_mean_cen)
plt.plot(LE_train_avr_cen)
plt.plot(RE_train_avr_cen)


comp_cen,info_cen,weights_cen = Do_PCA(Run01_0831_centered)
_ = Compoment_Visualize(comp_cen, all_cell_dic, r'C:\Users\ZR\Desktop\temp')

#%% use regressed train here
regressed_PCA_cen = PCA_Regression(comp_cen, info_cen, weights_cen,
                                   ignore_PC=[1],var_ratio = 0.75)
LE_cell_trains_reg = regressed_PCA_cen.loc[LE_cells]
LE_train_avr_reg = LE_cell_trains_reg.mean(0)
RE_cell_trains_reg = regressed_PCA_cen.loc[RE_cells]
RE_train_avr_reg = RE_cell_trains_reg.mean(0)
rest_cell_trains_reg = regressed_PCA_cen.loc[rest_cells]
rest_cell_reg_mean = rest_cell_trains_reg.mean(0)
all_cell_mean_reg = regressed_PCA_cen.loc[acn].mean(0)

#plt.plot(rest_cell_reg_mean)
plt.plot(all_cell_mean)
plt.plot(all_cell_mean_reg)


plt.plot(LE_train_avr_reg)
plt.plot(RE_train_avr_reg)
#%% shuffle, random select cells cannot get this

start_time = time.time()
times = 1000
LE_train_avr_shuffle = np.zeros(shape = (4971,times))
RE_train_avr_shuffle = np.zeros(shape = (4971,times))
corr_shuffle = np.zeros(times)
for i in range(times):
    c_rand_LE = random.sample(acn, 154)
    c_rand_RE = random.sample(lt.List_Subtraction(acn, c_rand_LE), 79)# make sure no overlap
    LE_train_avr_shuffle[:,i] = regressed_PCA_cen.loc[c_rand_LE].mean(0)
    RE_train_avr_shuffle[:,i] = regressed_PCA_cen.loc[c_rand_RE].mean(0)
    corr_shuffle[i],_ = stats.pearsonr(LE_train_avr_shuffle[:,i],RE_train_avr_shuffle[:,i])

end_time = time.time()
print('Time cost = %fs' % (end_time - start_time))

#%% Data struct transfer for seaborn plot. VERY SLOW
# =============================================================================
# base = Run01_0831_centered.copy()
# Frame_Num = Run01_0831_centered.shape[1]
# cell_num = Run01_0831_centered.shape[0]
# reshaped_cen_data = pd.DataFrame(index = range(cell_num*Frame_Num),columns = ['Cell_Name','Frame_Num','Value','Tuning'])
# for i,cc in enumerate(acn):
#     reshaped_cen_data.loc[i*Frame_Num:(i+1)*Frame_Num-1,'Cell_Name'] = cc
#     reshaped_cen_data.loc[i*Frame_Num:(i+1)*Frame_Num-1,'Value'] = np.array(base.loc[cc,:])
#     reshaped_cen_data.loc[i*Frame_Num:(i+1)*Frame_Num-1,'Frame_Num'] = list(range(Frame_Num))
#     
# for i in range(reshaped_cen_data.shape[0]):
#     if reshaped_cen_data.loc[i,'Cell_Name'] in LE_cells:
#         reshaped_cen_data.loc[i,'Tuning'] = 'LE'
#     elif reshaped_cen_data.loc[i,'Cell_Name'] in RE_cells:
#         reshaped_cen_data.loc[i,'Tuning'] = 'RE'
#     else:
#         reshaped_cen_data.loc[i,'Tuning'] = 'No_Tuning'
# =============================================================================
#%% Compare Pearson r results here.
from Stimulus_Cell_Processor.Cell_Info_Cross_Corr import Correlation_Core

work_folder = r'G:\_Pre_Processed_Data\210831_Loc18D_0.005-0.30'
T_Info_OD = ot.Load_Variable(work_folder,'T_info_Run06_OD.pkl')
T_values_OD = T_Info_OD['cell_info']
Map_LE = T_values_OD['L-0'].loc['t',:]
Map_RE = T_values_OD['R-0'].loc['t',:]
frame_num = 4971
LE_corr_plot = np.zeros(frame_num)
RE_corr_plot = np.zeros(frame_num)
for i in range(frame_num):
    c_frame = regressed_PCA_cen.iloc[:,i]
    LE_corr_plot[i],_ = Correlation_Core(Map_LE, c_frame)
    RE_corr_plot[i],_ = Correlation_Core(Map_RE, c_frame)
plt.plot(LE_corr_plot)
plt.plot(RE_corr_plot)
    
#%% Do T test for LE & RE -unregressed.
from tqdm import tqdm
LE_t = np.zeros(4971)
RE_t = np.zeros(4971)
for i in tqdm(range(4971)):
    c_LE = LE_cell_trains_cen.iloc[:,i]
    c_RE = RE_cell_trains_cen.iloc[:,i]
    c_null = rest_cell_trains_cen.iloc[:,i]
    _,_,LE_t[i] = T_Test_Pair(c_LE, c_null)
    _,_,RE_t[i] = T_Test_Pair(c_RE, c_null)
    

