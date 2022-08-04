# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 14:24:21 2022

@author: ZR

This script is used to generate Dist&Tuning regression on different time windows.

"""


from Series_Analyzer.Preprocessor_Cai import Pre_Processor_Cai
import OS_Tools_Kit as ot
from Series_Analyzer.Pairwise_Correlation import Series_Cut_Pair_Corr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm




#%% Initailization
wp = r'D:\ZR\_Temp_Data\220711_temp'

pcinfo76 = ot.Load_Variable(wp,'pc76_info.pkl')
pcwin76 = ot.Load_Variable(wp,'pc76win.pkl')
pcwin76_used = pcwin76.loc[:,60:]

pc_tuned = pcinfo76[pcinfo76.OD_A != -999]
pc_tuned = pc_tuned[pc_tuned.OD_B != -999]
pc_tuned = pc_tuned[pc_tuned.Orien_A != -999]
pc_tuned = pc_tuned[pc_tuned.Orien_B != -999]
tuned_lists = pc_tuned.index


tempwin = pcwin76_used.loc[tuned_lists,74].to_frame()
#%% We used OD diff, dist and other tuning to make correlation here.
tempwin['OD_diff'] = abs(pc_tuned['OD_A']-pc_tuned['OD_B'])
tempwin['Dist'] = pc_tuned['Dist']
tempwin = tempwin.rename(columns = {74:'Value'})

raw_orien_diff = abs(pc_tuned['Orien_A']-pc_tuned['Orien_B'])
raw_orien_diff[raw_orien_diff>90]=(180-raw_orien_diff)
tempwin['Orien_diff'] = raw_orien_diff

sns.lmplot(data = tempwin,x = 'Orien_diff',y = 'Value',scatter_kws = {'s':2})



X = tempwin['Dist']
Y = tempwin['Value']
X = sm.add_constant(X)
model = sm.OLS(Y,X)
result = model.fit()
result.summary()
X_fit = np.arange(0,90,0.2)
Y_fit = result.params[0]+result.params[1]*X_fit
# and kde scatter plot.
sns.kdeplot(data = tempwin, x = 'Dist',y = 'Value',fill=True,thresh=0, levels=100, cmap="rocket", color="w", linewidths=1)
sns.lineplot(X_fit,Y_fit,color = 'y')

#%% for each timewindow, get OD/Orien corr here.
window_ssa = pd.DataFrame(columns = ['SST','Dist','OD','Orien','Distr','ODr','Orienr'])

for i in range(pcwin76_used.shape[1]):
    pass

