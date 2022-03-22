# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 15:22:00 2022

@author: ZR

seperate cell by cell results using hue, check whether we get better data.
"""

import OS_Tools_Kit as ot
import cv2
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
from Series_Analyzer.Cell_Frame_PCA import Do_PCA,PCA_Regression
import matplotlib.pyplot as plt
from Series_Analyzer.Single_Component_Visualize import Single_Mask_Visualize
from Stimulus_Cell_Processor.Get_Tuning import Get_Tuned_Cells
import scipy.stats as stats
import numpy as np
import List_Operation_Kit as lt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
#%% read in 
day_folder = r'G:\Test_Data\2P\210831_L76_2P'
all_cell_tuning = ot.Load_Variable(day_folder+r'\All_Tuning_Property.tuning')
Run01_Frame = Pre_Processor(day_folder,'Run001',7000)
cell_by_cell_corr = ot.Load_Variable(day_folder+r'\Corr_Matrix.pkl')
acn =  list(Run01_Frame.index)
sig_thres = 0.01
LE_cells = []
RE_cells = []
No_Tune_cells = []
for i,cc in enumerate(acn):
    c_tuning = all_cell_tuning[cc]['LE']
    if c_tuning['t_value']>0 and c_tuning['p_value']<sig_thres:
        LE_cells.append(cc)
    elif c_tuning['t_value']<0 and c_tuning['p_value']<sig_thres:
        RE_cells.append(cc)
    else:
        No_Tune_cells.append(cc)
#%% annotate hue by cell diff
# Define flag, +1 as LE,-1 as RE,0 as no tune.
hued_cell_by_cell_corr = pd.DataFrame(columns = ['A_flag','B_flag','Corr','Tuning_Diff'])
for i in tqdm(range(len(cell_by_cell_corr))):
    c_pair = cell_by_cell_corr.loc[i,:]
    cell_A = c_pair['Cell_A']
    cell_B = c_pair['Cell_B']
    c_corr = c_pair['Pearsonr']
    c_td = c_pair['OD_Tuning_diff']
    if cell_A in LE_cells:
        c_A_flag = 1
    elif cell_A in RE_cells:
        c_A_flag = -1
    else:
        c_A_flag = 0
    if cell_B in LE_cells:
        c_B_flag = 1
    elif cell_B in RE_cells:
        c_B_flag = -1
    else:
        c_B_flag = 0
    hued_cell_by_cell_corr.loc[i] = [c_A_flag,c_B_flag,c_corr,c_td]
    
# calculate 
for i in tqdm(range(len(hued_cell_by_cell_corr))):
    c_pair = hued_cell_by_cell_corr.loc[i,:]
    multi = c_pair['A_flag']*c_pair['B_flag']
    add = c_pair['A_flag']+c_pair['B_flag']
    if multi>0:
        hued_cell_by_cell_corr.loc[i,'Type'] = 'Same'
    elif multi<0:
        hued_cell_by_cell_corr.loc[i,'Type'] = 'Different'
    elif multi ==0:
        if add ==0:
            hued_cell_by_cell_corr.loc[i,'Type'] = 'All_None'
        elif add != 0:
            hued_cell_by_cell_corr.loc[i,'Type'] = 'Single_None'
#%% plot graph by hue.
sns.lmplot(data = hued_cell_by_cell_corr,x = 'Tuning_Diff',y = 'Corr',hue = 'Type',scatter_kws = {'s':3,'alpha':0.5})    
#%% group data.
grouped_cell_by_cell_corr = list(hued_cell_by_cell_corr.groupby('Type'))
#same_sets = grouped_cell_by_cell_corr[2][1]
sns.lmplot(data = grouped_cell_by_cell_corr[0][1],x = 'Tuning_Diff',y = 'Corr',hue = 'Type',scatter_kws = {'s':3,'alpha':0.5})    


# get group data hists.
sns.histplot(data = grouped_cell_by_cell_corr[0][1],x = 'Corr',hue = 'Type',bins = 200)    
sns.histplot(data = grouped_cell_by_cell_corr[2][1],x = 'Corr',hue = 'Type',bins = 200) 

plt.hist(grouped_cell_by_cell_corr[0][1]['Corr'][0:3800],bins= 150,alpha = 0.6,color = 'r')
plt.hist(grouped_cell_by_cell_corr[2][1]['Corr'][3800:7600],bins= 150,alpha = 0.6,color = 'b')
	
# Do welch's t test to see differences.
# This shall use ttest_ind and equal_var=False
from scipy import stats
stats.ttest_ind(grouped_cell_by_cell_corr[0][1]['Corr'], grouped_cell_by_cell_corr[2][1]['Corr'], equal_var=False)



