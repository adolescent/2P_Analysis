# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 12:13:12 2022

@author: ZR

This part will generate reliable definition of network reactivation.

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
from scipy.stats import ttest_ind


wp = r'D:\ZR\_Temp_Data\220711_temp'

# Peak Find Lite.
from scipy.signal import find_peaks
def Peak_Finder(x,thres = 0):
    peaks, properties = find_peaks(x, height=thres,distance = 3,width = 0)
    plt.plot(x)
    plt.plot(np.zeros_like(x), "--", color="gray")
    #plt.plot(cell_response_frame['All_Num'])
    #plt.plot(cell_response_frame['LE_Num']/89)
    #plt.plot(cell_response_frame['RE_Num']/135)
    plt.plot(peaks, x[peaks], "x")
    plt.show()
    return peaks,properties


#%% Initialization same as New score system.
from Series_Analyzer.Response_Info import Get_Frame_Response

Run03_frame = Pre_Processor_Cai(r'D:\ZR\_Temp_Data\220420_L91',runname = 'Run003')
ot.Save_Variable(wp, r'Series91_Run03_0',Run03_frame)
peak_info = ot.Load_Variable(wp,'peak_info_91_before.pkl')
acd = ot.Load_Variable(wp,'All_Series_Dic91.pkl')
acinfo = ot.Load_Variable(wp,'Cell_Tuning_Dic91.pkl')
Run01_frame = ot.Load_Variable(wp,'Series91_Run01_0.pkl')
Run03_frame = Pre_Processor_Cai(r'D:\ZR\_Temp_Data\220420_L91',runname = 'Run003')
# ot.Save_Variable(wp, r'Series91_Run03_0',Run03_frame) # Already Done.
frame_response_before,cell_num_dic = Get_Frame_Response(Run01_frame,acinfo)
frame_response_after,_ = Get_Frame_Response(Run03_frame,acinfo)
peak_info['LE_prop'] = peak_info['LE_spike']/cell_num_dic['LE']
peak_info['RE_prop'] = peak_info['RE_spike']/cell_num_dic['RE']
peak_info['Orien0_prop'] = peak_info['Orien0_spike']/cell_num_dic['Orien0']
peak_info['Orien45_prop'] = peak_info['Orien45_spike']/cell_num_dic['Orien45']
peak_info['Orien90_prop'] = peak_info['Orien90_spike']/cell_num_dic['Orien90']
peak_info['Orien135_prop'] = peak_info['Orien135_spike']/cell_num_dic['Orien135']
#%% Get LE/RE peaks 
# This part discarded.
# =============================================================================
# LE_cell_name = tune_info[tune_info['OD']>0.5].index
# LE_trains = tuned_spikes.loc[LE_cell_name,:].mean()
# RE_cell_name = tune_info[tune_info['OD']<-0.5].index
# RE_trains = tuned_spikes.loc[RE_cell_name,:].mean()
# plt.plot(LE_trains)
# plt.plot(RE_trains)
# index =(LE_trains-RE_trains)
# plt.plot(index)
# 
# global_peak,global_peak_prop = Peak_Finder(tuned_spikes.mean(0),thres = 0.05)
# LE_peak,LE_peak_prop = Peak_Finder(LE_trains,thres = 0.1)
# RE_peak,RE_peak_prop = Peak_Finder(RE_trains,thres = 0.1)
# # cycle all LE peak to see nearby RE peak.
# Eye_peak_diff = []
# for i,cp in enumerate(LE_peak):
#     min_dist = 9999
#     for j,cp_ref in enumerate(global_peak):
#         c_dist = abs(cp-cp_ref)
#         if c_dist<min_dist:
#             min_dist = c_dist
#     Eye_peak_diff.append(min_dist)
# 
# plt.hist(Eye_peak_diff,bins = range(0,41,1))
# =============================================================================

#%% Compare whether LE RE fires have difference.
#LE_fires = LE_trains[LE_peak]
#RE_fires = RE_trains[RE_peak]
#plt.hist(LE_fires,bins = 50,alpha = 0.75)
#plt.hist(RE_fires,bins = 50,alpha = 0.75)

plt.hist(peak_info['LE_Prop'],bins = 20,alpha = 0.75)
plt.hist(peak_info['RE_Prop'],bins = 20,alpha = 0.75)
t,p = ttest_ind(peak_info['LE_Prop'],peak_info['RE_Prop'],equal_var = False) # Do Welch's Test
# get LE/RE index of each peak, see whether they are unbaised.
index = []
for i in peak_info.index:
    c_peak = peak_info.loc[i]
    if (c_peak['LE_spike']+c_peak['RE_spike'])>0:# not a 0-0 event
        c_index =(c_peak['LE_Prop']-c_peak['RE_Prop'])/(c_peak['LE_Prop']+c_peak['RE_Prop'])
        index.append(c_index)

#%% Compare before & After series 
after_peak,after_peak_info = Peak_Finder(frame_response_after['All_Num'],thres = 10)
peak_info_after = frame_response_after.loc[after_peak,:]


LE_after = peak_info_after['LE_prop']
RE_after = peak_info_after['RE_prop']
ttest_ind(LE_after, RE_after,equal_var = False)
plt.hist(LE_after,bins=list(np.arange(0,4,0.2)),alpha = 0.75)
plt.hist(RE_after,bins=list(np.arange(0,4,0.2)),alpha = 0.75)



