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

#%% 
for i in range(frame_num):
    c_frame = Run01_Frame.iloc[:,i]
    OD_corr_plot[i],_ = Correlation_Core(Map_OD, c_frame)
    LE_corr_plot[i],_ = Correlation_Core(Map_LE, c_frame)
    RE_corr_plot[i],_ = Correlation_Core(Map_RE, c_frame)
plt.plot(LE_corr_plot)
plt.plot(RE_corr_plot)
#plt.plot(OD_corr_plot)
#%% Get most correlated frames.
from Series_Analyzer.Single_Component_Visualize import Single_Comp_Visualize
from Timecourse_Tools.Most_Correlated_Frames import Most_Correlated_Index

all_cell_dic = ot.Load_Variable(r'G:\Test_Data\2P\210831_L76_2P\L76_210831A_All_Cells.ac')

LE_highest = Most_Correlated_Index(LE_corr_plot,prop = 0.1)
LE_lowest = Most_Correlated_Index(LE_corr_plot,mode = 'Low',prop = 0.1)
LE_highest_avr = Run01_Frame.iloc[:,LE_highest].mean(1)
LE_lowest_avr = Run01_Frame.iloc[:,LE_lowest].mean(1)
LE_highest_avr_graph = Single_Comp_Visualize(all_cell_dic,LE_highest_avr)
LE_lowest_avr_graph = Single_Comp_Visualize(all_cell_dic,LE_lowest_avr)


RE_highest = Most_Correlated_Index(RE_corr_plot,prop = 0.1)
RE_lowest = Most_Correlated_Index(RE_corr_plot,mode = 'Low',prop = 0.1)
RE_highest_avr = Run01_Frame.iloc[:,RE_highest].mean(1)
RE_lowest_avr = Run01_Frame.iloc[:,RE_lowest].mean(1)
RE_highest_avr_graph = Single_Comp_Visualize(all_cell_dic,RE_highest_avr)
RE_lowest_avr_graph = Single_Comp_Visualize(all_cell_dic,RE_lowest_avr)
