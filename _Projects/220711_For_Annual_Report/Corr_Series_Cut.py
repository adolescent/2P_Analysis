# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 17:24:36 2022

@author: ZR

"""

from Series_Analyzer.Preprocessor_Cai import Pre_Processor_Cai
import OS_Tools_Kit as ot
from Series_Analyzer.Pairwise_Correlation import Series_Cut_Pair_Corr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr




#%% Initailization
wp = r'D:\ZR\_Temp_Data\220711_temp'

series76 = Pre_Processor_Cai(r'D:\ZR\_Temp_Data\220630_L76_2P',start_frame=0,runname = 'Run001')
series85 = Pre_Processor_Cai(r'D:\ZR\_Temp_Data\220706_L85_LM',start_frame=0,runname = 'Run001')
series91 = Pre_Processor_Cai(r'D:\ZR\_Temp_Data\220420_L91',start_frame=0,runname = 'Run001')

ot.Save_Variable(wp, 'Series_76_Run1', series76)
ot.Save_Variable(wp, 'Series_85_Run1', series85)
ot.Save_Variable(wp, 'Series_91_Run1', series91)

acd76 = ot.Load_Variable(r'D:\ZR\_Temp_Data\220630_L76_2P\_CAIMAN','All_Series_Dic.pkl')
tuning_dic76 = ot.Load_Variable(r'D:\ZR\_Temp_Data\220630_L76_2P\_CAIMAN','Cell_Tuning_Dic.pkl')
pcinfo76,pcwin76 = Series_Cut_Pair_Corr(acd76, tuning_dic76,series76,win_size = 300,win_step = 60)
ot.Save_Variable(wp, 'pc76_info', pcinfo76)
ot.Save_Variable(wp, 'pc76win', pcwin76)


acd85 = ot.Load_Variable(r'D:\ZR\_Temp_Data\220706_L85_LM\_CAIMAN','All_Series_Dic.pkl')
tuning_dic85 = ot.Load_Variable(r'D:\ZR\_Temp_Data\220706_L85_LM\_CAIMAN','Cell_Tuning_Dic.pkl')
pcinfo85,pcwin85 = Series_Cut_Pair_Corr(acd85, tuning_dic85,series85,win_size = 300,win_step = 60)
ot.Save_Variable(wp, 'pc85_info', pcinfo85)
ot.Save_Variable(wp, 'pc85win', pcwin85)


acd91 = ot.Load_Variable(r'D:\ZR\_Temp_Data\220420_L91\_CAIMAN','All_Series_Dic.pkl')
tuning_dic91 = ot.Load_Variable(r'D:\ZR\_Temp_Data\220420_L91\_CAIMAN','Cell_Tuning_Dic.pkl')
pcinfo91,pcwin91 = Series_Cut_Pair_Corr(acd91, tuning_dic91,series91,win_size = 300,win_step = 60)
ot.Save_Variable(wp, 'pc91_info', pcinfo91)
ot.Save_Variable(wp, 'pc91win', pcwin91)


#%% Plot corr distribution of L76
winnum = pcwin76.shape[1]
all_bins = np.zeros(shape = (50,winnum),dtype = 'f8')
scale = np.linspace(-0.5,1,50) # scale of hist.
bin76 = pd.DataFrame(columns = ['Corr','Freq','winnum'],index = range(winnum*50))
counter = 0
for i in range(winnum):
    c_series = pcwin76.loc[:,i]
    c_bins = plt.hist(c_series,bins = 50,range = (-0.5,1))[0]
    all_bins[:,i] = c_bins
    for j in range(50):
        bin76.loc[counter] = [c_bins[j],scale[j],i]
        counter +=1
bin76 = bin76.astype('f8')
# start from 60min to avoid scramble frame.
bin76 = bin76.round({'Freq':3,'Corr':0,'winnum':0})
a = bin76.pivot(index = 'Freq',columns = 'winnum',values = 'Corr')
a = a.loc[:,60:]
sns.heatmap(a)
avr76 = pcwin76_used.mean(0)
#%% we divide cells into different groups.
pcwin76_used = pcwin76[:,60:]
LE_pairs = pcinfo76[pcinfo76['OD_A']>0.5][pcinfo76['OD_B']>0.5]
LE_pairs_ind = LE_pairs.index
LE_corrs76 = pcwin76_used.loc[LE_pairs_ind,:]
LE_corrs76.columns.name = 'Window'
LE_corrs76.index.name = 'Index'
a = LE_corrs76.stack().to_frame()
a = a.rename({0: 'Value'}, axis='columns')
sns.lineplot(data = a,x = 'Window',y = 'Value')
# get RE pairs.
RE_pairs = pcinfo76[pcinfo76['OD_A']<-0.5][pcinfo76['OD_B']<-0.5]
RE_pairs_ind = RE_pairs.index
RE_corrs76 = pcwin76_used.loc[RE_pairs_ind,:]
RE_corrs76.columns.name = 'Window'
RE_corrs76.index.name = 'Index'
b = RE_corrs76.stack().to_frame()
b = b.rename({0: 'Value'}, axis='columns')
sns.lineplot(data = b,x = 'Window',y = 'Value')


# plot them together.
fig,ax = plt.subplots()
sns.lineplot(data = a,x = 'Window',y = 'Value')
sns.lineplot(data = b,x = 'Window',y = 'Value')
plt.plot(avr76)

LE_avr = LE_corrs76.mean()
RE_avr = RE_corrs76.mean()
pearsonr(LE_avr,RE_avr)




