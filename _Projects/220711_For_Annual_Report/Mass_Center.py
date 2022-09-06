# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:57:35 2022

@author: ZR

This script will calcuate weighted dist std of each ensemble events.

"""

from Series_Analyzer.Preprocessor_Cai import Pre_Processor_Cai
import OS_Tools_Kit as ot
from Series_Analyzer.Pairwise_Correlation import Series_Cut_Pair_Corr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr,spearmanr
import statsmodels.api as sm
import pandas as pd
from tqdm import tqdm
from Series_Analyzer.Series_Cutter import Series_Window_Slide
import cv2

wp = r'D:\ZR\_Temp_Data\220711_temp'
tuned_series = ot.Load_Variable(wp,'Tuned_Spikes_91.pkl')
acd = ot.Load_Variable(wp,'All_Series_Dic91.pkl')
peak_info = ot.Load_Variable(wp,'peak_info_91.pkl')
#%% 
peak_frames = tuned_series[peak_info.index]

#%% define mass center calculator.
def Mass_Center_Calculator(input_frame,acd):
    active_cells = input_frame[input_frame>0]
    active_cell_id = active_cells.index
    act_num = len(active_cell_id)
    total_act = active_cells.sum()
    sum_y = 0
    sum_x = 0
    for i,cc in enumerate(active_cell_id):
        ccy,ccx = acd[cc]['Cell_Loc']
        sum_y += active_cells[cc]*ccy
        sum_x += active_cells[cc]*ccx
    center_y = sum_y/total_act
    center_x = sum_x/total_act
    # calculate ss
    ss =0
    for i,cc in enumerate(active_cell_id):
        ccy,ccx = acd[cc]['Cell_Loc']
        c_x_diff = np.square(ccx-center_x)
        c_y_diff = np.square(ccy-center_y)
        c_ss = c_x_diff+c_y_diff
        ss += c_ss
        
    sd = np.sqrt(ss/total_act)

    return center_y,center_x,act_num,ss,sd

#%% calculate each peak 
sd_frame = pd.DataFrame(0,columns = ['sd','cell_num'],index = peak_frames.columns)
for i,cp in enumerate(peak_frames.columns):
    c_peak = peak_frames.loc[:,cp]
    _,_,_,_,c_st = Mass_Center_Calculator(c_peak, acd)
    c_cell_num = peak_info.loc[cp,'All_Num']
    sd_frame.loc[cp,:] = [c_st,c_cell_num]
sd_frame['cellnum_group'] = sd_frame['cell_num']//50
a = sd_frame.groupby('cellnum_group').mean()
