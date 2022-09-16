# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 13:23:55 2022

@author: ZR

This part will calculate response of different animals.

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

#%% re generate L85 & L76 series.
series76 = Pre_Processor_Cai(r'D:\ZR\_Temp_Data\220630_L76_2P',start_frame = 4000)
ot.Save_Variable(wp, 'Series76_Run01_4000', series76)
series85 = Pre_Processor_Cai(r'D:\ZR\_Temp_Data\220706_L85_LM',start_frame = 3000)
ot.Save_Variable(wp, 'Series85_Run01_3000', series85)
#%% Initailization
acd = ot.Load_Variable(wp,'All_Series_Dic91.pkl')
acinfo = ot.Load_Variable(wp,'Cell_Tuning_Dic91.pkl')
dataframe1 = ot.Load_Variable(wp,'Series_91_Run1.pkl')
spikes = dataframe1[dataframe1>2]
spikes = spikes.fillna(0).clip(lower = -5,upper = 5)
sns.heatmap(spikes,center = 0,vmax = 5)
#%% get tuning frames.
actune = pd.DataFrame(columns = ['OD','Orien'])
acn = list(acd.keys())
for i,cc in enumerate(acn):
    tc = acinfo[cc]
    if tc['Fitted_Orien'] != 'No_Tuning':
        actune.loc[cc] = [tc['OD']['Tuning_Index'],tc['Fitted_Orien']]
        
# =============================================================================
# od_sorted_index = actune.sort_values('OD').index
# orien_sorted_index = actune.sort_values('Orien').index
# od_sorted_frame = spikes.reindex(od_sorted_index)
# orien_sorted_frame = spikes.reindex(orien_sorted_index)
# sns.heatmap(od_sorted_frame,center = 0)
# =============================================================================
# average every 50 lines.
#a = od_sorted_frame.groupby(np.arange(len(od_sorted_frame))//50).mean()
# visualize spike change.
tuned_spikes = spikes.reindex(actune.index)
cellnum,framenum = tuned_spikes.shape
tuned_cell = list(tuned_spikes.index)
#%% Calculate each frame sumed network response.
# 1. No tuning weight ver.
spike_info_column = ['All_Num','All_spike','LE_Num','LE_spike','RE_Num','RE_spike','Orien0_Num','Orien0_spike','Orien45_Num','Orien45_spike','Orien90_Num','Orien90_spike','Orien135_Num','Orien135_spike']
# Use 0,45,90,135+-22.5 as group order.
actune['Orien_Group'] = (((actune['Orien']+22.5)%180)//45)+1
cell_response_frame = pd.DataFrame(0.0,columns=spike_info_column,index = range(framenum))
OD_thres = 0.5
LE_cell_Num = (actune['OD']>OD_thres).sum()
RE_cell_Num = (actune['OD']<-OD_thres).sum()
Orien0_cell_Num = (actune['Orien_Group']==1).sum()
Orien45_cell_Num = (actune['Orien_Group']==2).sum()
Orien90_cell_Num = (actune['Orien_Group']==3).sum()
Orien135_cell_Num = (actune['Orien_Group']==4).sum()
for i in tqdm(range(framenum)):
    #cframe = tuned_spikes.loc[:,i]
    cframe = tuned_spikes.loc[:,i]
    firing_cells = cframe[cframe>0]
    cell_response_frame.loc[i,'All_Num'] = len(firing_cells)
    cell_response_frame.loc[i,'All_spike'] = firing_cells.sum()
    fire_cell_names = list(firing_cells.index)
    # then cycle 
    for j,cc in enumerate(fire_cell_names):
        c_tune = actune.loc[cc,:]
        if c_tune['OD']>OD_thres: # LE cell
            cell_response_frame.loc[i,'LE_Num'] +=1
            cell_response_frame.loc[i,'LE_spike'] += firing_cells[cc]
        elif c_tune['OD']<-OD_thres: # RE cell
            cell_response_frame.loc[i,'RE_Num'] +=1
            cell_response_frame.loc[i,'RE_spike'] += firing_cells[cc]
        # orien counter
        if c_tune['Orien_Group'] == 1 : # Orien 0 cell
            cell_response_frame.loc[i,'Orien0_Num'] +=1
            cell_response_frame.loc[i,'Orien0_spike'] += firing_cells[cc]
        elif c_tune['Orien_Group'] == 2 : # Orien 45 cell
            cell_response_frame.loc[i,'Orien45_Num'] +=1
            cell_response_frame.loc[i,'Orien45_spike'] += firing_cells[cc]
        elif c_tune['Orien_Group'] == 3 : # Orien 90 cell
            cell_response_frame.loc[i,'Orien90_Num'] +=1
            cell_response_frame.loc[i,'Orien90_spike'] += firing_cells[cc]
        elif c_tune['Orien_Group'] == 4 : # Orien 135 cell
            cell_response_frame.loc[i,'Orien135_Num'] +=1
            cell_response_frame.loc[i,'Orien135_spike'] += firing_cells[cc]
    
# find peak
from scipy.signal import find_peaks
x = cell_response_frame['All_spike']
peaks, properties = find_peaks(x, height=10,distance = 3,width = 0)
plt.plot(x)
plt.plot(np.zeros_like(x), "--", color="gray")
#plt.plot(cell_response_frame['All_Num'])
#plt.plot(cell_response_frame['LE_Num']/89)
#plt.plot(cell_response_frame['RE_Num']/135)
plt.plot(peaks, x[peaks], "x")
plt.show()



