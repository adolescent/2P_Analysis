# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 12:31:37 2022

@author: ZR

This script count cell response of every single frame. Counting single cell spon resposne.

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

#%% Initailization
acd = ot.Load_Variable(wp,'All_Series_Dic85.pkl')
acinfo = ot.Load_Variable(wp,'Cell_Tuning_Dic85.pkl')
dataframe1 = ot.Load_Variable(wp,'Series_85_Run1.pkl')
spikes = dataframe1[dataframe1>2]
spikes = spikes.fillna(0).clip(upper = 5)
sns.heatmap(spikes,center = 0,vmax = 5)
#%% reindex frame 
# get tuning frames.
actune = pd.DataFrame(columns = ['OD','Orien'])
acn = list(acd.keys())
for i,cc in enumerate(acn):
    tc = acinfo[cc]
    if tc['Fitted_Orien'] != 'No_Tuning':
        actune.loc[cc] = [tc['OD']['Tuning_Index'],tc['Fitted_Orien']]
        
od_sorted_index = actune.sort_values('OD').index
orien_sorted_index = actune.sort_values('Orien').index
od_sorted_frame = spikes.reindex(od_sorted_index)
orien_sorted_frame = spikes.reindex(orien_sorted_index)
sns.heatmap(od_sorted_frame,center = 0)
# average every 50 lines.
#a = od_sorted_frame.groupby(np.arange(len(od_sorted_frame))//50).mean()
# visualize spike change.
tuned_spikes = spikes.reindex(actune.index)
cellnum,framenum = tuned_spikes.shape
tuned_cell = list(tuned_spikes.index)
all_run01_frame = np.zeros(shape = (512,512,framenum))
for i,cc in tqdm(enumerate(tuned_cell)):
    ccy,ccx = acd[cc]['Cell_Loc']
    ccy = int(ccy)
    ccx = int(ccx)
    for j in range(framenum):
        c_strengh = float(tuned_spikes.loc[cc,j])
        if c_strengh !=0:
            all_run01_frame[:,:,j] = cv2.circle(img = np.float32(all_run01_frame[:,:,j]),center = (ccx,ccy),radius = 5,color = c_strengh,thickness = -1)
            
            
#ot.Save_Variable(wp, 'L91_Spike_Video_Run01', all_run01_frame)

# Write video.
data_for_write = (all_run01_frame*51).astype('u1')
fps = 8
graph_size = (512,512)
video_writer = cv2.VideoWriter(wp+r'\\L85_Run01_Video.mp4',cv2.VideoWriter_fourcc('X','V','I','D'),fps,graph_size,0)
for i in range(framenum):
    u1_writable_graph = data_for_write[:,:,i]
    annotated_graph = cv2.putText(u1_writable_graph.astype('f4'),'Stim ID = '+str(i),(250,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,255,1)
    video_writer.write(annotated_graph.astype('u1'))
del video_writer 


