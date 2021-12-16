# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 10:21:08 2021

@author: ZR
"""
#%% Build data for spon analyze.
from My_Wheels.Series_Analyzer import Spontaneous_Preprocessing as Prepro
from Series_Analyzer.Cell_Frame_PCA import Do_PCA,Compoment_Visualize
import matplotlib.pyplot as plt
from Series_Analyzer.Cell_Activity_Evaluator import Pre_Processed_Data_Count
import seaborn as sns
import OS_Tools_Kit as ot
from Analyzer.My_FFT import FFT_Power

#%% Processor
save_folder = r'G:\_Pre_Processed_Data\210604_Loc18B_0.005-0.30'
processed_data = Prepro.Pre_Processor(r'G:\Test_Data\2P\210604_L76_2P',runname = 'Run013',
                                      start_time = 0,passed_band = (0.005,0.3),order = 7)

#%% Testor, use once a time.
counter,_ = Pre_Processed_Data_Count(processed_data)
act = counter.mean(0)
#%% Savior
ot.Save_Variable(save_folder, 'Run13_383cell_All_Spon_After', processed_data)



