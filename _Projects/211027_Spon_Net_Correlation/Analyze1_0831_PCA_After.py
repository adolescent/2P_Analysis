# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 13:20:40 2021

@author: ZR
"""

import OS_Tools_Kit as ot
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
from Series_Analyzer.Cell_Frame_PCA import Do_PCA,Compoment_Visualize
from Stimulus_Cell_Processor.Map_Tuning_Calculator import PC_Tuning_Calculation
from Stimulus_Cell_Processor.Cell_Info_Cross_Corr import PC_Comp_vs_t_Maps
import pandas as pd
import Analyzer.My_FFT as FFT
import matplotlib.pyplot as plt
import seaborn as sns

from Series_Analyzer.TC_Analyze import Peak_Counter
import statsmodels.api as sm

#%% Generate cell frame and do PCA first.
all_t_graph = ot.Load_Variable(r'G:\Test_Data\2P\210831_L76_2P\_All_Results\All_Stim_t_Graph.pkl')
day_folder = r'G:\Test_Data\2P\210831_L76_2P'
all_cell_dic = ot.Load_Variable(day_folder,'L76_210831A_All_Cells.ac')
cell_series_after = Pre_Processor(day_folder,'Run003')
# make sure we use same cells.
before_pc = ot.Load_Variable(r'G:\_Processed_Results\211027_Spon_Net_Correlation\PCA_Results\0831_Before_PCA\PC_Components.pkl')
used_cell = before_pc.index
used_after_series = cell_series_after.loc[used_cell]
components,PCA_info,fitted_weights = Do_PCA(used_after_series)
output_folder = r'G:\_Processed_Results\211027_Spon_Net_Correlation\PCA_Results\PCA_Graphs'
_ = Compoment_Visualize(components, all_cell_dic, output_folder)
ot.Save_Variable(output_folder, 'PC_Components',components)
ot.Save_Variable(output_folder, 'PC_info',PCA_info)
ot.Save_Variable(output_folder, 'fitted_weights',fitted_weights)
# Calculate PC tunings.
PC_Tunings,PC_Tuning_Matrix = PC_Tuning_Calculation(components,day_folder)
ot.Save_Variable(output_folder, 'PCA_tunings', PC_Tunings)
PC_stim_Frames,PC_stim_p,PC_judge = PC_Comp_vs_t_Maps(components,all_t_graph)




#%% PCA Time Course
before_PCA_weights = ot.Load_Variable(r'G:\_Processed_Results\211027_Spon_Net_Correlation\PCA_Results\0831_Before_PCA','fitted_weights.pkl')
PC1_expt = before_PCA_weights.loc[:,'PC001'].tolist()


PC1_before = before_PCA_weights.loc[:,'PC001']
spectrum = FFT.FFT_Power(PC1_before)
spectrum_windowed = FFT.FFT_Window_Slide(PC1_before,window_length=120)

PC3_before = before_PCA_weights.loc[:,'PC003']
spectrum = FFT.FFT_Power(PC3_before)
spectrum_windowed = FFT.FFT_Window_Slide(PC3_before,window_length=120)

PC3_after = after_PCA_weights.loc[:,'PC003']
spectrum = FFT.FFT_Power(PC3_after)
spectrum_windowed = FFT.FFT_Window_Slide(PC3_after,window_length=120)

PC6_before = before_PCA_weights.loc[:,'PC006']
spectrum = FFT.FFT_Power(PC6_before)
spectrum_windowed = FFT.FFT_Window_Slide(PC6_before,window_length=120)


#%% Auto correlation of plots.

auto_corr = sm.tsa.acf(PC6_after,nlags = 100)

#%% Count PC components peak repeat before/after

before_peak_series = Peak_Counter(PC3_before,win_size=120,win_step = 30)
after_peak_series = Peak_Counter(PC3_after,win_size=120,win_step = 30)
OD_compare_Frame = pd.DataFrame(columns = ['Before','After'])
OD_compare_Frame.loc[:,'After'] = after_peak_series
OD_compare_Frame.loc[0:116,'Before'] = before_peak_series
plt.plot(OD_compare_Frame)

before_orien_series = Peak_Counter(PC6_before,win_size=120,win_step = 30)
after_orien_series = Peak_Counter(PC6_after,win_size=120,win_step = 30)
Orien_compare_Frame = pd.DataFrame(columns = ['Before','After'])
Orien_compare_Frame.loc[:,'After'] = after_orien_series
Orien_compare_Frame.loc[0:116,'Before'] = before_orien_series
plt.plot(Orien_compare_Frame)
#%% Plot PC count 
PCA_weight_before = ot.Load_Variable(r'G:\_Processed_Results\211027_Spon_Net_Correlation\PCA_Results\0831_Before_PCA\fitted_weights.pkl')
all_PC_comp = ot.Load_Variable(r'G:\_Processed_Results\211027_Spon_Net_Correlation\PCA_Results\0831_Before_PCA\PC_Components.pkl')
used_PCA_weight_Before = PCA_weight_Before.iloc[:,2:12].T
used_PC_name = used_PCA_weight_Before.index.tolist()

Spike_Count_Before = pd.DataFrame(columns = used_PC_name)
for i,c_PC in enumerate(used_PC_name):
    c_count_series = Peak_Counter(used_PCA_weight_Before.loc[c_PC],win_size=120,win_step = 30)
    Spike_Count_Before.loc[:,c_PC] = c_count_series
    
used_PCA_weight_After = PCA_weight_After.iloc[:,2:12].T
Spike_Count_After = pd.DataFrame(columns = used_PC_name)
for i,c_PC in enumerate(used_PC_name):
    c_count_series = Peak_Counter(used_PCA_weight_After.loc[c_PC],win_size=120,win_step = 30)
    Spike_Count_After.loc[:,c_PC] = c_count_series
#%% Vector Autoregression model
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

VAR_Data_Before = used_PCA_weight_Before.T
model_Before = VAR(VAR_Data_Before)
results = model_Before.fit(maxlags=5, ic='aic')
a = results.summary()

#Calculate long-term state change.
VAR_Data_Before_Windowed = Spike_Count_Before
model_Before_Windowed = 


