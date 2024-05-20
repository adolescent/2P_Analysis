# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 16:27:49 2022

@author: ZR

New shuffle reserve space distribution of cell groups

"""

import OS_Tools_Kit as ot
import numpy as np
from Stimulus_Cell_Processor.Cell_Info_Cross_Corr import Correlation_Core
import matplotlib.pyplot as plt
from Series_Analyzer.Single_Component_Visualize import Single_Comp_Visualize
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor


work_folder = r'G:\_Pre_Processed_Data\210831_Loc18D_0.005-0.30'
Run01_Frame = Pre_Processor(r'G:\Test_Data\2P\210831_L76_2P',start_time=7000)
T_Info_OD = ot.Load_Variable(work_folder,'T_info_Run06_OD.pkl')
T_values_OD = T_Info_OD['cell_info']
Map_OD = T_values_OD['OD'].loc['t',:]
Map_LE = T_values_OD['L-0'].loc['t',:]
Map_RE = T_values_OD['R-0'].loc['t',:]
frame_num = Run01_Frame.shape[1]

#%% get LE/RE corr plot here.
OD_corr_plot = np.zeros(frame_num)
LE_corr_plot = np.zeros(frame_num)
RE_corr_plot = np.zeros(frame_num)
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

#%% Read in shuffled graphs
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

shuffle1 = cv2.imread(r'C:\Users\ZR\Desktop\masks\1.bmp',-1)
mask = shuffle1.astype('bool')
all_cell_dic = ot.Load_Variable(r'G:\Test_Data\2P\210831_L76_2P\L76_210831A_All_Cells.ac')
acn = list(Run01_Frame.index)
all_cell_info = np.zeros(shape = (len(acn),2),dtype = 'i4')
# get all cell loc here.
for i,cc in tqdm(enumerate(acn)):
    cc_loc = all_cell_dic[cc]['Cell_Info'].centroid
    all_cell_info[i,0] = cc_loc[0]
    all_cell_info[i,1] = cc_loc[1]
    
# find cell in/out masks

cell_in_mask = []
cell_out_mask = []
for i,cc in tqdm(enumerate(acn)):
    c_loc = all_cell_info[i,:]
    if mask[c_loc[0],c_loc[1]] == True:
        cell_in_mask.append(cc)
    elif mask[c_loc[0],c_loc[1]] == False:
        cell_out_mask.append(cc)
        
cell_in_frame = pd.DataFrame(index = cell_in_mask)
cell_in_frame['Value'] = 1
cell_out_frame = pd.DataFrame(index = cell_out_mask)
cell_out_frame['Value'] = 1
Single_Comp_Visualize(all_cell_dic, cell_in_frame)
Single_Comp_Visualize(all_cell_dic, cell_out_frame)

#%% Generate PC1 regressed series here.
from Series_Analyzer.Cell_Frame_PCA import Do_PCA,PCA_Regression
import matplotlib.pyplot as plt
import scipy.stats as stats
comp,info,weights = Do_PCA(Run01_Frame)
regressed_PCA = PCA_Regression(comp, info, weights,
                               ignore_PC = [1],var_ratio = 0.75)

Shuffle_Cells_LE = random.sample(cell_in_mask, 20)
Shuffle_Cells_RE = random.sample(cell_out_mask, 20)

Shuffle_Cells_LE = random.sample(Real_LE_cells, 20)
Shuffle_Cells_RE = random.sample(Real_RE_cells, 20)

Shuffle_LE_trains = regressed_PCA.loc[Shuffle_Cells_LE,:]
Shuffle_RE_trains = regressed_PCA.loc[Shuffle_Cells_RE,:]
Shuffle_LE_mean = Shuffle_LE_trains.mean(0)
Shuffle_RE_mean = Shuffle_RE_trains.mean(0)
plt.plot(Shuffle_LE_mean)
plt.plot(Shuffle_RE_mean)
stats.pearsonr(Shuffle_LE_mean,Shuffle_RE_mean)

#%% correct tuning infos
ac_tunings = ot.Load_Variable(r'G:\Test_Data\2P\210831_L76_2P\\','All_Tuning_Property.tuning')
Shuffle_LE_map = pd.DataFrame(index = Shuffle_Cells_LE)
Shuffle_LE_map['Value'] = 1
Shuffle_RE_map = pd.DataFrame(index = Shuffle_Cells_RE)
Shuffle_RE_map['Value'] = 1
from Stimulus_Cell_Processor.Map_Tuning_Calculator import Map_Tuning_Core
Shuffle_LE_tunings = Map_Tuning_Core(ac_tunings, Shuffle_LE_map)
Shuffle_RE_tunings = Map_Tuning_Core(ac_tunings, Shuffle_RE_map)
    
#%% Get real LE/RE cells
thres = 0.01
Real_LE_cells = []
Real_RE_cells = []
for i,cc in enumerate(acn):
    c_tuning_OD = ac_tunings[cc]['LE']
    if (c_tuning_OD['t_value']>0 and c_tuning_OD['p_value']<thres):
        Real_LE_cells.append(cc)
    elif (c_tuning_OD['t_value']<0 and c_tuning_OD['p_value']<thres):
        Real_RE_cells.append(cc)
        
        
        