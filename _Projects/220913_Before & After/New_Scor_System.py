# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:21:49 2022

@author: ZR
Input peak information, then we generate score to find most likely peaks.

Work will be as followed.(If Orien)
1.get prop toward avr 4 orien networks
2.define score weight. e.g.:For Orien0, use orien0 as 1,orien 90 as -1, orien45/135 as -0.5
3.Calculate score and rating.
4.Regenerate most active 50/100/200 peaks.

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

wp = r'F:\_Data_Temp\220711_temp'

#%% read in and get cell info.
peak_info = ot.Load_Variable(wp,'peak_info_91.pkl')
acd = ot.Load_Variable(wp,'All_Series_Dic91.pkl')
acinfo = ot.Load_Variable(wp,'Cell_Tuning_Dic91.pkl')
dataframe1 = ot.Load_Variable(wp,'Series_91_Run1.pkl')
spikes = dataframe1[dataframe1>2]
spikes = spikes.fillna(0).clip(lower = -5,upper = 5)
OD_thres = 0.5
acn = list(acd.keys())
tune_info = pd.DataFrame(columns = ['OD','Orien'])
for i,cc in tqdm(enumerate(acn)):
    ccd = acinfo[cc]
    #if ccd['OD_Preference'] != 'No_Tuning':
    if ccd['Orien_Preference'] != 'No_Tuning':
        tune_info.loc[cc,:] = [ccd['OD']['Tuning_Index'],ccd['Fitted_Orien']]
tune_info = tune_info.astype('f8')
tune_info['Orien_Group'] = (((tune_info['Orien']+22.5)%180)//45)*45
tuned_spikes = spikes.loc[tune_info.index]
LE_num = (tune_info['OD']>0.5).sum()
RE_num = (tune_info['OD']<-0.5).sum()
Orien0_num = (tune_info['Orien_Group'] == 0).sum()
Orien45_num = (tune_info['Orien_Group'] == 45).sum()
Orien90_num = (tune_info['Orien_Group'] == 90).sum()
Orien135_num = (tune_info['Orien_Group'] == 135).sum()
# get different network firing props.
peak_info['LE_Prop'] = peak_info['LE_Num']/LE_num
peak_info['RE_Prop'] = peak_info['RE_Num']/RE_num
peak_info['Orien0_Prop'] = peak_info['Orien0_Num']/Orien0_num
peak_info['Orien45_Prop'] = peak_info['Orien45_Num']/Orien45_num
peak_info['Orien90_Prop'] = peak_info['Orien90_Num']/Orien90_num
peak_info['Orien135_Prop'] = peak_info['Orien135_Num']/Orien135_num

#%% Get new tuning scores.
# test on Orien 0 cells.
peak_info['Orien0_Score'] = peak_info['Orien0_Prop']-peak_info['Orien90_Prop']-0.5*peak_info['Orien45_Prop']-0.5*peak_info['Orien135_Prop']
peak_info['Orien90_Score'] = peak_info['Orien90_Prop']-peak_info['Orien0_Prop']-0.5*peak_info['Orien45_Prop']-0.5*peak_info['Orien135_Prop']
peak_info['Orien45_Score'] = peak_info['Orien45_Prop']-peak_info['Orien135_Prop']-0.5*peak_info['Orien0_Prop']-0.5*peak_info['Orien90_Prop']
peak_info['Orien135_Score'] = peak_info['Orien135_Prop']-peak_info['Orien45_Prop']-0.5*peak_info['Orien0_Prop']-0.5*peak_info['Orien90_Prop']
peak_info['LE_Score'] = peak_info['LE_Prop'] - peak_info['RE_Prop']
peak_info['RE_Score'] = peak_info['RE_Prop'] - peak_info['LE_Prop']


#%% and regenerate graphs.
Best_peak = peak_info.sort_values('RE_Score',ascending=False)
#Best_peak = peak_info[(peak_info['Orien0_Num']>5)*(peak_info['Single_ON']== True)]
#LE_only_peak = peak_info.sort_values('RE_spike',ascending=False)
restore = tuned_spikes.loc[:,Best_peak.index[:100]].mean(1)
restore_map = np.zeros(shape = (512,512))
for i,cc in enumerate(restore.index):
    ccy,ccx = acd[cc]['Cell_Loc']
    ccy = int(ccy)
    ccx = int(ccx)
    restore_map = cv2.circle(img = np.float32(restore_map),center = (ccx,ccy),radius = 4,color = restore[cc],thickness = -1)
    
sns.heatmap(restore_map,center = 0,square = True,xticklabels=False,yticklabels=False)





