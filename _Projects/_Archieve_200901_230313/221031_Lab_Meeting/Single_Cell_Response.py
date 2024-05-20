# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 13:10:46 2022

@author: ZR

For generation of single cell resposne.
"""

import OS_Tools_Kit as ot
import matplotlib.pyplot as plt
from Series_Analyzer.Preprocessor_Cai import Pre_Processor_Cai
import pandas as pd
from tqdm import tqdm

#%% choose a run of anthes change.
tamplate_acd = ot.Load_Variable(r'G:\Test_Data\2P\220810_L85_2P\_CAIMAN\All_Ceries_Dic_Washed.pkl')
# get dF/F trains of each cell.
a  = Pre_Processor_Cai(r'G:\Test_Data\2P\220810_L85_2P',runname = 'Run001')

#%% compare dF/F before and after.
L91_cd = ot.Load_Variable(r'D:\ZR\_Temp_Data\220420_L91\_CAIMAN','All_Series_Dic.pkl')
from My_Wheels.Filters import Signal_Filter
def CD2Spike(cell_dic,runname='1-001'):
    acn = list(cell_dic.keys())
    frame_num = len(cell_dic[acn[0]][runname])
    spike_train = pd.DataFrame(0,columns = acn,index = range(frame_num))
    for cc in tqdm(acn):
        tc = cell_dic[cc][runname]
        c_train = (tc-tc.mean())/tc.mean()
        c_train = Signal_Filter(c_train,filter_para = (0.01/1.301,0.6/1.301),order = 7)
        spike_train[cc] = c_train
    spike_train = spike_train.T
    spike_train = spike_train.astype('f8')
    return spike_train

train_spon = CD2Spike(L91_cd,runname = '1-001')
train_stim = CD2Spike(L91_cd,runname = '1-007')
plt.plot(train_spon.mean(0))
plt.plot(train_stim.mean(0))
#%% Single cell show
sc_spon = train_spon.loc[23,7000:].reset_index(drop = 1)
sc_stim = train_stim.loc[23,:].reset_index(drop = 1)
plt.plot(sc_spon)
plt.plot(sc_stim)
#%% Get max dF/F of each cell.
from scipy.stats import ttest_rel
import numpy as np
spon_max = train_spon.max(1)
stim_max = train_stim.max(1)

plt.hist(spon_max,bins = (np.arange(0,2,0.05)))
plt.hist(stim_max,bins = (np.arange(0,2,0.05)))
ttest_rel(spon_max, stim_max)
#%% Get cell firing rate compare.
spon_std = train_spon.std(1)
stim_std = train_stim.std(1)
spike_spon = pd.DataFrame(0,columns = train_spon.index,index = train_spon.columns)
spike_stim = pd.DataFrame(0,columns = train_stim.index,index = train_stim.columns)
for i in tqdm(range(651)):
    c_tran = train_spon.iloc[i,:]
    c_std = spon_std.iloc[i]
    fire_train = c_tran*(c_tran>2*c_std)
    spike_spon[i+1] = fire_train
    # stim
    c_tran = train_stim.iloc[i,:]
    c_std = stim_std.iloc[i]
    fire_train = c_tran*(c_tran>2*c_std)
    spike_stim[i+1] = fire_train
    
plt.plot(spike_stim.iloc[0,:])
fr_stim = spike_stim.mean(0)
fr_spon = spike_spon.iloc[8000:,:].mean(0)

plt.hist(fr_spon,bins = (np.arange(0,0.1,0.004)))
plt.hist(fr_stim,bins = (np.arange(0,0.1,0.004)))
ttest_rel(fr_spon, fr_stim)
#%%

pca_info_before = ot.Load_Variable(r'D:\ZR\_Temp_Data\220630_L76_2P\_CAIMAN\Spon_Before_PCA','All_PC_Info.pkl')



