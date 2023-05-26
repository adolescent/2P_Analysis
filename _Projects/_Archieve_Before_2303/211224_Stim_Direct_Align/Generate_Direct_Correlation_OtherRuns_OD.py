# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 15:21:40 2022

@author: ZR

"""

import OS_Tools_Kit as ot
import numpy as np
from Stimulus_Cell_Processor.Cell_Info_Cross_Corr import Correlation_Core
import matplotlib.pyplot as plt
from Series_Analyzer.Single_Component_Visualize import Single_Comp_Visualize
from Stimulus_Cell_Processor.T_Map_Generator import One_Key_T_Maps
import scipy.stats as stats
#%% First,generate OD T Map information.
all_t_info_0721 = One_Key_T_Maps(r'G:\Test_Data\2P\210721_L76_2P','Run006',runtype = 'OD_2P')
all_t_info_0920 = One_Key_T_Maps(r'G:\Test_Data\2P\210920_L76_2P','Run006',runtype = 'OD_2P')



#%% Read in all information.
all_cell_dic_0721 = ot.Load_Variable(r'G:\Test_Data\2P\210721_L76_2P\L76_210721A_All_Cell_Include_Run03.ac')
work_folder_0721 = r'G:\_Pre_Processed_Data\210721_LocSM_0.005-0.30'
Run01_Frame_0721 = ot.Load_Variable(work_folder_0721,'Run01_280cell_3100-All_Spon_Before.pkl')
T_Info_OD_0721 = ot.Load_Variable(work_folder_0721,'Run06_OD_T_Info.pkl')
T_values_OD_0721 = T_Info_OD_0721['cell_info']
Map_OD_0721 = T_values_OD_0721['OD'].loc['t',:]
Map_LE_0721 = T_values_OD_0721['L-0'].loc['t',:]
Map_RE_0721 = T_values_OD_0721['R-0'].loc['t',:]


all_cell_dic_0920 = ot.Load_Variable(r'G:\Test_Data\2P\210920_L76_2P\L76_210920A_All_Cells.ac')
work_folder_0920 = r'G:\_Pre_Processed_Data\210920_Loc18B_0.005-0.30'
Run01_Frame_0920 = ot.Load_Variable(work_folder_0920,'Run01_377cell_6000s-All_Spon_Before.pkl')
T_values_OD_0920 = ot.Load_Variable(work_folder_0920,r'Run06_OD_T_Info.pkl.pkl')['cell_info']
Map_OD_0920 = T_values_OD_0920['OD'].loc['t',:]
Map_LE_0920 = T_values_OD_0920['L-0'].loc['t',:]
Map_RE_0920 = T_values_OD_0920['R-0'].loc['t',:]
#%% Calculate corr plots.
LE_corr_plot_0721 = np.zeros(10037)
RE_corr_plot_0721 = np.zeros(10037)
for i in range(10037):
    c_frame = Run01_Frame_0721.iloc[:,i]
    LE_corr_plot_0721[i],_ = Correlation_Core(Map_LE_0721, c_frame)
    RE_corr_plot_0721[i],_ = Correlation_Core(Map_RE_0721, c_frame)
plt.plot(LE_corr_plot_0721)
plt.plot(RE_corr_plot_0721)

LE_corr_plot_0920 = np.zeros(6262)
RE_corr_plot_0920 = np.zeros(6262)
for i in range(6262):
    c_frame = Run01_Frame_0920.iloc[:,i]
    LE_corr_plot_0920[i],_ = Correlation_Core(Map_LE_0920, c_frame)
    RE_corr_plot_0920[i],_ = Correlation_Core(Map_RE_0920, c_frame)
plt.plot(LE_corr_plot_0920)
plt.plot(RE_corr_plot_0920)

#%% Shuffle 
# Shuffle 0721 1000 times
import scipy.stats as stats
import time
import pandas as pd
time_start = time.time()
times = 1000
corr_values_0721 = np.zeros(times)
correlation_plots_LE_0721 = np.zeros(shape = (10037,times))
correlation_plots_RE_0721 = np.zeros(shape = (10037,times))
for i in range(times):
    origin_graph = pd.DataFrame([Map_LE_0721,Map_RE_0721]).T
    shuffled_data = origin_graph.sample(frac = 1,axis = 0).reset_index(drop = True)
    shuffled_data.index = list(origin_graph.index)
    shuffled_LE = shuffled_data.iloc[:,0]
    shuffled_RE = shuffled_data.iloc[:,1]
    shuffled_LE_corr_plot = np.zeros(10037)
    shuffled_RE_corr_plot = np.zeros(10037)
    for j in range(10037):
        c_frame = Run01_Frame_0721.iloc[:,j]
        shuffled_LE_corr_plot[j],_ = Correlation_Core(shuffled_LE, c_frame)
        shuffled_RE_corr_plot[j],_ = Correlation_Core(shuffled_RE, c_frame)
    c_r,_ = stats.pearsonr(shuffled_LE_corr_plot,shuffled_RE_corr_plot)
    corr_values_0721[i] = c_r
    correlation_plots_LE_0721[:,i] = shuffled_LE_corr_plot
    correlation_plots_RE_0721[:,i] = shuffled_RE_corr_plot
time_end = time.time()
print('Time cost = %fs' % (time_end - time_start))
ot.Save_Variable(work_folder_0721, 'LE_shuffled_plots', correlation_plots_LE_0721)
ot.Save_Variable(work_folder_0721, 'RE_shuffled_plots', correlation_plots_RE_0721)

# Shuffle Data 0920
time_start = time.time()
times = 1000
corr_values_0920 = np.zeros(times)
correlation_plots_LE_0920 = np.zeros(shape = (6262,times))
correlation_plots_RE_0920 = np.zeros(shape = (6262,times))
for i in range(times):
    origin_graph = pd.DataFrame([Map_LE_0721,Map_RE_0721]).T
    shuffled_data = origin_graph.sample(frac = 1,axis = 0).reset_index(drop = True)
    shuffled_data.index = list(origin_graph.index)
    shuffled_LE = shuffled_data.iloc[:,0]
    shuffled_RE = shuffled_data.iloc[:,1]
    shuffled_LE_corr_plot = np.zeros(6262)
    shuffled_RE_corr_plot = np.zeros(6262)
    for j in range(6262):
        c_frame = Run01_Frame_0721.iloc[:,j]
        shuffled_LE_corr_plot[j],_ = Correlation_Core(shuffled_LE, c_frame)
        shuffled_RE_corr_plot[j],_ = Correlation_Core(shuffled_RE, c_frame)
    c_r,_ = stats.pearsonr(shuffled_LE_corr_plot,shuffled_RE_corr_plot)
    corr_values_0920[i] = c_r
    correlation_plots_LE_0920[:,i] = shuffled_LE_corr_plot
    correlation_plots_RE_0920[:,i] = shuffled_RE_corr_plot
time_end = time.time()
print('Time cost = %fs' % (time_end - time_start))
ot.Save_Variable(work_folder_0920, 'LE_shuffled_plots', correlation_plots_LE_0920)
ot.Save_Variable(work_folder_0920, 'RE_shuffled_plots', correlation_plots_RE_0920)
#%% Plot Corr 
corr_values_0920 = np.zeros(1000)
for i in range(1000):
    c_r,_ = stats.pearsonr(correlation_plots_LE_0920[:,i],correlation_plots_RE_0920[:,i])
    corr_values_0920[i] = c_r
    
    
    