# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 16:12:48 2021

@author: ZR
"""

import OS_Tools_Kit as ot
import numpy as np
from Stimulus_Cell_Processor.Cell_Info_Cross_Corr import Correlation_Core
import matplotlib.pyplot as plt
from Series_Analyzer.Single_Component_Visualize import Single_Comp_Visualize

work_folder = r'G:\_Pre_Processed_Data\210831_Loc18D_0.005-0.30'
Run01_Frame = ot.Load_Variable(work_folder,'Run01_300cell_7000-All_Spon_Before.pkl')
T_Info_OD = ot.Load_Variable(work_folder,'T_info_Run06_OD.pkl')
T_values_OD = T_Info_OD['cell_info']
Map_OD = T_values_OD['OD'].loc['t',:]
Map_LE = T_values_OD['L-0'].loc['t',:]
Map_RE = T_values_OD['R-0'].loc['t',:]
frame_num = Run01_Frame.shape[1]

OD_corr_plot = np.zeros(frame_num)
LE_corr_plot = np.zeros(frame_num)
RE_corr_plot = np.zeros(frame_num)

#%% get LE/RE corr plot here.
for i in range(frame_num):
    c_frame = Run01_Frame.iloc[:,i]
    OD_corr_plot[i],_ = Correlation_Core(Map_OD, c_frame)
    LE_corr_plot[i],_ = Correlation_Core(Map_LE, c_frame)
    RE_corr_plot[i],_ = Correlation_Core(Map_RE, c_frame)
plt.plot(LE_corr_plot)
plt.plot(RE_corr_plot)
#plt.plot(OD_corr_plot)

normed_RE_plot = RE_corr_plot/RE_corr_plot.max()
plt.plot(normed_RE_plot)
plt.plot(a)
#%% Get most correlated frames.
from Series_Analyzer.Single_Component_Visualize import Single_Comp_Visualize
from Timecourse_Tools.Most_Correlated_Frames import Most_Correlated_Index

all_cell_dic = ot.Load_Variable(r'G:\Test_Data\2P\210831_L76_2P\L76_210831A_All_Cells.ac')

LE_highest = Most_Correlated_Index(LE_corr_plot,prop = 0.05)
LE_lowest = Most_Correlated_Index(LE_corr_plot,mode = 'Low',prop = 0.05)
LE_highest_avr = Run01_Frame.iloc[:,LE_highest].mean(1)
LE_lowest_avr = Run01_Frame.iloc[:,LE_lowest].mean(1)
LE_highest_avr_graph = Single_Comp_Visualize(all_cell_dic,LE_highest_avr)
LE_lowest_avr_graph = Single_Comp_Visualize(all_cell_dic,LE_lowest_avr)


RE_highest = Most_Correlated_Index(RE_corr_plot,prop = 0.05)
RE_lowest = Most_Correlated_Index(RE_corr_plot,mode = 'Low',prop = 0.05)
RE_highest_avr = Run01_Frame.iloc[:,RE_highest].mean(1)
RE_lowest_avr = Run01_Frame.iloc[:,RE_lowest].mean(1)
RE_highest_avr_graph = Single_Comp_Visualize(all_cell_dic,RE_highest_avr)
RE_lowest_avr_graph = Single_Comp_Visualize(all_cell_dic,RE_lowest_avr)
#%% Shuffle
import pandas as pd
origin_graph = pd.DataFrame([Map_LE,Map_RE]).T
shuffled_data = origin_graph.sample(frac = 1,axis = 0).reset_index(drop = True)
shuffled_data.index = list(origin_graph.index)
shuffled_LE = shuffled_data.iloc[:,0]
shuffled_RE = shuffled_data.iloc[:,1]

shuffled_LE_corr_plot = np.zeros(frame_num)
shuffled_RE_corr_plot = np.zeros(frame_num)
for i in range(frame_num):
    c_frame = Run01_Frame.iloc[:,i]
    shuffled_LE_corr_plot[i],_ = Correlation_Core(shuffled_LE, c_frame)
    shuffled_RE_corr_plot[i],_ = Correlation_Core(shuffled_RE, c_frame)
    
stats.pearsonr(shuffled_LE_corr_plot,shuffled_RE_corr_plot)
    
plt.plot(LE_corr_plot)
plt.plot(shuffled_LE_corr_plot)
plt.plot(RE_corr_plot)
plt.plot(shuffled_RE_corr_plot)
#plt.plot(OD_corr_plot)



#%% Do above on regressed data
LE_corr_plot_regressed = np.zeros(frame_num)
RE_corr_plot_regressed = np.zeros(frame_num)
for i in range(frame_num):
    c_frame = Run01_Frame_regressed.iloc[:,i]
    LE_corr_plot_regressed[i],_ = Correlation_Core(Map_LE, c_frame)
    RE_corr_plot_regressed[i],_ = Correlation_Core(Map_RE, c_frame)
    
    
plt.plot(LE_corr_plot_regressed)
plt.plot(RE_corr_plot_regressed)

shuffled_LE_corr_plot_regressed = np.zeros(frame_num)
shuffled_RE_corr_plot_regressed = np.zeros(frame_num)
for i in range(frame_num):
    c_frame = Run01_Frame_regressed.iloc[:,i]
    shuffled_LE_corr_plot_regressed[i],_ = Correlation_Core(shuffled_LE, c_frame)
    shuffled_RE_corr_plot_regressed[i],_ = Correlation_Core(shuffled_RE, c_frame)
    
plt.plot(LE_corr_plot_regressed)
plt.plot(shuffled_LE_corr_plot_regressed)
plt.plot(RE_corr_plot_regressed)
plt.plot(shuffled_RE_corr_plot_regressed)


plt.plot(shuffled_LE_corr_plot_regressed)
plt.plot(shuffled_RE_corr_plot_regressed)




#%% 1000 shuffles, run it over night.
import scipy.stats as stats
import time
time_start = time.time()



times = 1000
corr_values = np.zeros(times)
correlation_plots_LE = np.zeros(shape = (4971,times))
correlation_plots_RE = np.zeros(shape = (4971,times))
for i in range(times):
    
    origin_graph = pd.DataFrame([Map_LE,Map_RE]).T
    shuffled_data = origin_graph.sample(frac = 1,axis = 0).reset_index(drop = True)
    shuffled_data.index = list(origin_graph.index)
    shuffled_LE = shuffled_data.iloc[:,0]
    shuffled_RE = shuffled_data.iloc[:,1]
    
    shuffled_LE_corr_plot = np.zeros(frame_num)
    shuffled_RE_corr_plot = np.zeros(frame_num)
    for j in range(frame_num):
        c_frame = Run01_Frame.iloc[:,j]
        shuffled_LE_corr_plot[j],_ = Correlation_Core(shuffled_LE, c_frame)
        shuffled_RE_corr_plot[j],_ = Correlation_Core(shuffled_RE, c_frame)
    c_r,_ = stats.pearsonr(shuffled_LE_corr_plot,shuffled_RE_corr_plot)
    corr_values[i] = c_r
    correlation_plots_LE[:,i] = shuffled_LE_corr_plot
    correlation_plots_RE[:,i] = shuffled_RE_corr_plot
    

time_end = time.time()
print('Time cost = %fs' % (time_end - time_start))

save_folder = r'G:\_Pre_Processed_Data\210831_Loc18D_0.005-0.30'
ot.Save_Variable(save_folder, 'LE_shuffled_plots', correlation_plots_LE)
ot.Save_Variable(save_folder, 'RE_shuffled_plots', correlation_plots_RE)

#%% Plot Raster map of L-0/R-0
# First, get the threshold.
thres = 1.5 # std of threshold
LE_thres = LE_corr_plot.mean()+LE_corr_plot.std()*thres
RE_thres = RE_corr_plot.mean()+RE_corr_plot.std()*thres
id_above_thres_LE = np.where(LE_corr_plot>LE_thres)[0]
thres_LE_avr = Run01_Frame.iloc[:,id_above_thres_LE].mean(1)
LE_highest_avr_graph = Single_Comp_Visualize(all_cell_dic,thres_LE_avr)

id_above_thres_RE = np.where(RE_corr_plot>RE_thres)[0]
thres_RE_avr = Run01_Frame.iloc[:,id_above_thres_RE].mean(1)
RE_highest_avr_graph = Single_Comp_Visualize(all_cell_dic,thres_RE_avr)

plt.plot(LE_corr_plot)
plt.plot(RE_corr_plot)

# Then, use this thres to generate event series.
LE_rasters = LE_corr_plot>LE_thres
RE_rasters = RE_corr_plot>RE_thres
all_events = LE_rasters+RE_rasters

labels = ['No_OD','LE','RE','Both']
value = [4287,288,393,3]
plt.pie(value, labels = labels)
labels = ['LE','RE','Both']
value = [288,393,3]
plt.pie(value, labels = labels)
#%% Control shuffled result.
thres = 1.5 # std of threshold
common_num = np.zeros(1000)
common_prop = np.zeros(1000)
for i in range(1000):
    c_LE_shuf = correlation_plots_LE[:,i]
    c_RE_shuf = correlation_plots_RE[:,i]
    c_LE_thres = c_LE_shuf.mean()+c_LE_shuf.std()*thres
    c_RE_thres = c_LE_shuf.mean()+c_RE_shuf.std()*thres
    c_LE_rast = (c_LE_shuf>c_LE_thres)
    c_RE_rast = (c_RE_shuf>c_RE_thres)
    c_all = (c_LE_rast+c_RE_rast).sum()
    c_common = (c_LE_rast*c_RE_rast).sum()
    common_num[i] = c_common
    common_prop[i] = c_common/c_all

#%% Calculate cross correlation 
# move RE along LE
RE_kernel = RE_corr_plot[:3971]
LE_tar_series = LE_corr_plot
RE_along_LE_cc = np.correlate(LE_tar_series, RE_kernel)

LE_kernel = LE_corr_plot[:3971]
RE_tar_series = RE_corr_plot
LE_along_RE_cc = np.correlate(RE_tar_series, LE_kernel)


plt.plot(RE_along_LE_cc)
plt.plot(LE_along_RE_cc)

# vs shuffle
shuffled_RE_along_LE = np.zeros(shape = (1000,1001))
shuffled_LE_along_RE = np.zeros(shape = (1000,1001))
for i in range(1000):
    c_LE_tar = shuffle_correlation_plots_LE[:,i]
    c_RE_tar = shuffle_correlation_plots_RE[:,i]
    c_RE_ker = shuffle_correlation_plots_RE[:3971,i]
    c_LE_ker = shuffle_correlation_plots_LE[:3971,i]
    shuffled_RE_along_LE[i,:] = np.correlate(c_LE_tar,c_RE_ker)
    shuffled_LE_along_RE[i,:] = np.correlate(c_RE_tar,c_LE_ker)
    
    
a = shuffled_RE_along_LE[104,:]
b = shuffled_LE_along_RE[104,:]
plt.plot(a)
plt.plot(b)

#%% Get Granger causality
# Get windowed data structure first.
window_size = 300
win_step = 60
fps = 1.301

win_frame = int(300*fps)
step_frame = int(win_step*fps)
win_num = (4971-win_frame)//step_frame+1 # Ignore last one.
all_cutted_LE_corr = np.zeros(shape = (win_frame,win_num))
all_cutted_RE_corr = np.zeros(shape = (win_frame,win_num))

for i in range(win_num):
    all_cutted_LE_corr[:,i] = LE_corr_plot[i*step_frame:i*step_frame+win_frame]
    all_cutted_RE_corr[:,i] = LE_corr_plot[i*step_frame:i*step_frame+win_frame]
