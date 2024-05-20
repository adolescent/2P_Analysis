# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 13:12:57 2021

@author: ZR
"""
#%%
from My_Wheels.Series_Analyzer import Spontaneous_Preprocessing as Prepro
import matplotlib.pyplot as plt
from Analyzer.My_FFT import FFT_Power
import OS_Tools_Kit as ot
import pandas as pd

example_day = r'G:\Test_Data\2P\210831_L76_2P'

raw_data = Prepro.Pre_Processor(example_day,runname = 'Run001',
                                start_time = 0,passed_band = (False,False),order = 7)
before_data = Prepro.Pre_Processor(example_day,runname = 'Run001',
                                   start_time = 0,passed_band = (0.05,0.5),order = 2)
processed_data = Prepro.Pre_Processor(example_day,runname = 'Run001',
                                      start_time = 0,passed_band = (0.005,0.3),order = 7)

cell_raw = raw_data.iloc[25,:]
cell_before = before_data.iloc[25,:]
cell_after = processed_data.iloc[25,:]
raw_pow = FFT_Power(cell_raw)
before_pow = FFT_Power(cell_before)
after_pow = FFT_Power(cell_after)

#%% Plot part
plt.plot(cell_raw,alpha = 0.8)
plt.plot(cell_before,alpha = 0.8)
plt.plot(cell_after,alpha = 0.8)
#%% Plot Frequency
plt.plot(raw_pow,alpha = 0.7)
plt.plot(before_pow,alpha = 0.7)
plt.plot(after_pow,alpha = 0.7)
#%% PCA Test Before & After
from Series_Analyzer.Cell_Frame_PCA import Do_PCA,Compoment_Visualize
all_cell_dic = ot.Load_Variable(r'G:\Test_Data\2P\210831_L76_2P\L76_210831A_All_Cells.ac')
comp_after,info_after,weight_after = Do_PCA(processed_data.iloc[:,9367:])
comp_raw,info_raw,weight_raw = Do_PCA(raw_data.iloc[:,9367:])
comp_before,info_before,weight_before = Do_PCA(before_data.iloc[:,9367:])

PC_Graph_Raw = Compoment_Visualize(comp_raw, all_cell_dic, r'G:\Test_Data\2P\210831_L76_2P\_All_Results')
PC_Graph_Before = Compoment_Visualize(comp_before, all_cell_dic, r'G:\Test_Data\2P\210831_L76_2P\_All_Results')
PC_Graph_After = Compoment_Visualize(comp_after, all_cell_dic, r'G:\Test_Data\2P\210831_L76_2P\_All_Results')

#%% Before PCA Freqs
used_pc = list(weight_after.columns)[0:15]
all_pc_fft = pd.DataFrame()
for i,cpc in enumerate(used_pc):
    c_weight = weight_after.loc[:,cpc]
    c_pow = FFT_Power(c_weight,normalize = False)
    all_pc_fft[cpc] = c_pow
#%% Do binned fft power
for i in range(0,2356):
    all_pc_fft.iloc[i,-1] = i//20
a = all_pc_fft.copy()
a = a.groupby('Group')
a_sum = a.sum()
