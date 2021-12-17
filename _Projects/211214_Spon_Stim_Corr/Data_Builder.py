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
save_folder = r'G:\_Pre_Processed_Data\200115_L80_Loc01_0.005-0.30'
processed_data = Prepro.Pre_Processor(r'G:\Test_Data\2P\200115_L80_LM_Loc1',runname = 'Run014',
                                      start_time = 0,fps = 1.301,passed_band = (0.005,0.3),order = 7)

#%% Testor, use once a time.
counter,_ = Pre_Processed_Data_Count(processed_data,fps = 1.301)
act = counter.mean(0)
#%% Savior
ot.Save_Variable(save_folder, 'Run14_233cell_0s-All_Spon_After', processed_data)



#%%
from My_Wheels.Standard_Cell_Generator import Standard_Cell_Generator

Scg = Standard_Cell_Generator('L80', '210115', r'G:\Test_Data\2P\200115_L80_LM_Loc1',[9,11,12,13,14]
                              ,cell_subfolder=r'\_Morpho_Cells',location = 'Loc1')
Scg.Generate_Cells()
#%% Cell producer
from My_Wheels.Cell_Find_From_Graph import Cell_Find_And_Plot
morpho_cells = Cell_Find_And_Plot(r'G:\Test_Data\2P\200115_L80_LM_Loc1','Global_Average.tif',
                                  'Morpho',find_thres=1.25)
#%%
from My_Wheels.Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'G:\Test_Data\2P\200115_L80_LM_Loc1\200115_L80_2P_stimuli')