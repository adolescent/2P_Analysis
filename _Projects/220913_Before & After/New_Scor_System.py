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

wp = r'D:\ZR\_Temp_Data\220711_temp'

#%% read in and get cell info.
peak_info = ot.Load_Variable(wp,'peak_info_91.pkl')
acd = ot.Load_Variable(wp,'All_Series_Dic91.pkl')
acinfo = ot.Load_Variable(wp,'Cell_Tuning_Dic91.pkl')
dataframe1 = ot.Load_Variable(wp,'Series_91_Run1.pkl')
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
LE_num = (tune_info['OD']>0.5).sum()
RE_num = (tune_info['OD']<-0.5).sum()
Orien0_num = (tune_info['Orien_Group'] == 0).sum()
Orien45_num = (tune_info['Orien_Group'] == 45).sum()
Orien90_num = (tune_info['Orien_Group'] == 90).sum()
Orien135_num = (tune_info['Orien_Group'] == 135).sum()


#%% Get new tuning scores.





