# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:25:15 2022

@author: ZR
"""

import OS_Tools_Kit as ot
import cv2
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
from Series_Analyzer.Cell_Frame_PCA import Do_PCA,PCA_Regression
import matplotlib.pyplot as plt
from Series_Analyzer.Single_Component_Visualize import Single_Mask_Visualize
from Stimulus_Cell_Processor.Get_Tuning import Get_Tuned_Cells
import scipy.stats as stats
import numpy as np

#%% read in 
day_folder = r'G:\Test_Data\2P\210831_L76_2P'
all_cell_dic = ot.Load_Variable(day_folder,'L76_210831A_All_Cells.ac')
Run01_Frame = Pre_Processor(day_folder,start_time = 7000)
acn = list(Run01_Frame.index)




