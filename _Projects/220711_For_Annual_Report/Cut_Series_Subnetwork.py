# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 13:08:30 2022

@author: ZR
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


#%% Initailization
wp = r'D:\ZR\_Temp_Data\220711_temp'
pcinfo91 = ot.Load_Variable(wp,'pc91_info.pkl')
pcwin91 = ot.Load_Variable(wp,'pc91win.pkl')

pc_tuned = pcinfo91[pcinfo91.OD_A != -999]
pc_tuned = pc_tuned[pc_tuned.OD_B != -999]
pc_tuned = pc_tuned[pc_tuned.Orien_A != -999]
pc_tuned = pc_tuned[pc_tuned.Orien_B != -999]
tuned_lists = pc_tuned.index
pcinfo91_used = pcinfo91.loc[tuned_lists,:]
OD_diff = abs(pcinfo91_used['OD_A']-pcinfo91_used['OD_B'])
raw_orien_diff = abs(pcinfo91_used['Orien_A']-pcinfo91_used['Orien_B'])
raw_orien_diff[raw_orien_diff>90]=(180-raw_orien_diff)
Orien_diff = raw_orien_diff
pcinfo91_used['OD_diff'] = OD_diff
pcinfo91_used['Orien_diff'] = Orien_diff
ot.Save_Variable(wp, 'pc91tuned_info', pcinfo91_used)
pcwin91_tuned = pcwin91.loc[tuned_lists,:]
ot.Save_Variable(wp, 'pc91win_tuned', pcwin91_tuned)

#%% get subnetworks.
pcinfo91t = ot.Load_Variable(wp,'pc91tuned_info.pkl')
pcwin91t = ot.Load_Variable(wp,'pc91win_tuned.pkl')
avr91 = pcwin91t.mean()
# similar eyes
RE_pairs = pcinfo91t[pcinfo91t['OD_A']<-0.5][pcinfo91t['OD_B']<-0.5]
LE_pairs = pcinfo91t[pcinfo91t['OD_A']>0.5][pcinfo91t['OD_B']>0.5]
sameeye_pairs = pd.concat([RE_pairs,LE_pairs])
sameeye_index = sameeye_pairs.index
sameeye_win = pcwin91t.loc[sameeye_index,:]
avr_sameeye = sameeye_win.mean()
# similar orientation
sameorien_pairs = pcinfo91t[pcinfo91t['Orien_diff']<22.5]
sameorien_index = sameorien_pairs.index
sameorien_win = pcwin91t.loc[sameorien_index,:]
avr_sameorien = sameorien_win.mean()



plt.plot(avr_sameeye-avr91)
plt.plot(avr_sameorien-avr91)
plt.plot(avr91)

# need shuffle to see it's not random results.
time = 2000
A_matrix = np.zeros(shape = (2000,187))
B_matrix = np.zeros(shape = (2000,187))
r_lists = []

for i in tqdm(range(time)):
    samp1 = pcwin91t.sample(20000).mean()-avr91
    samp2 = pcwin91t.sample(20000).mean()-avr91
    A_matrix[i,:] = samp1
    B_matrix[i,:] = samp2
    cr,_ = spearmanr(samp1[30:140],samp2[30:140])
    r_lists.append(cr)
    


#%% Then seperate orien,od network here.
from Series_Analyzer.Pair_Corr_Tools import Win_Corr_Select
LE_index = LE_pairs.index
RE_index = RE_pairs.index
LE_win = pcwin91t.loc[LE_index,:]
RE_win = pcwin91t.loc[RE_index,:]
LR_info = Win_Corr_Select((pcinfo91t['OD_A']>0.5)*(pcinfo91t['OD_B']<-0.5), pcwin91t)
LR_info2 = Win_Corr_Select((pcinfo91t['OD_A']<-0.5)*(pcinfo91t['OD_B']>0.5), pcwin91t)
LR_info = pd.concat([LR_info,LR_info2]).index
LR_win = pcwin91t.loc[LR_info,:]
# Seperate orientation diff into 4 group.
pcinfo91t['Orien_group'] = pcinfo91t['Orien_diff']//22.5+1
#orien_groups = dict(list(pcinfo91t.groupby('Orien_group'))) #Use function.
orien1_win = Win_Corr_Select(pcinfo91t['Orien_group']==1, pcwin91t)
orien2_win = Win_Corr_Select(pcinfo91t['Orien_group']==2, pcwin91t)
orien3_win = Win_Corr_Select(pcinfo91t['Orien_group']==3, pcwin91t)
orien4_win = Win_Corr_Select(pcinfo91t['Orien_group']==4, pcwin91t)

plt.plot(orien1_win.mean())
plt.plot(orien2_win.mean())
plt.plot(orien3_win.mean())
plt.plot(orien4_win.mean())
plt.plot(avr91)

plt.plot(LE_win.mean())
plt.plot(RE_win.mean())
plt.plot(LR_win.mean())
plt.plot(avr91)

plt.plot(LE_win.mean()-avr91)
plt.plot(RE_win.mean()-avr91)
plt.plot(LR_win.mean()-avr91)
plt.plot(A_matrix.mean(0))

plt.plot(orien1_win.mean()-avr91)
plt.plot(orien2_win.mean()-avr91)
plt.plot(orien3_win.mean()-avr91)
plt.plot(orien4_win.mean()-avr91)



#%% Do ther same thing on L76 & L85 data.
