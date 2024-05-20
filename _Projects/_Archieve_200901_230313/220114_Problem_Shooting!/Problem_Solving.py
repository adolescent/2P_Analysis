# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 16:09:17 2022

@author: ZR

Direct correlation & Global Regression Problem

"""
import OS_Tools_Kit as ot
import numpy as np
from Stimulus_Cell_Processor.Cell_Info_Cross_Corr import Correlation_Core
import matplotlib.pyplot as plt
from Series_Analyzer.Single_Component_Visualize import Single_Comp_Visualize,Single_Mask_Visualize
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
import pandas as pd
from tqdm import tqdm
import scipy.stats as stats
import seaborn as sns
from Stimulus_Cell_Processor.Get_Tuning import Get_Tuned_Cells
import List_Operation_Kit as lt
import random
from Analyzer.Statistic_Tools import T_Test_Pair
#%% Part1 Correlation Matrix
day_folder = r'G:\Test_Data\2P\210831_L76_2P'
all_cell_dic = ot.Load_Variable(day_folder,r'L76_210831A_All_Cells.ac')


Run01_Frame =  Pre_Processor(day_folder,start_time=7000)
acn = list(Run01_Frame.index)
correlation_matrix = pd.DataFrame(index = acn,columns = acn)

for i,cell_A in tqdm(enumerate(acn)):
    for j,cell_B in enumerate(acn):
        c_r,_ = stats.pearsonr(Run01_Frame.loc[cell_A,:],Run01_Frame.loc[cell_B,:])
        correlation_matrix.loc[cell_A,cell_B] = c_r
correlation_matrix = correlation_matrix.fillna(0)
LE_cells = list(set(Get_Tuned_Cells(day_folder, 'LE'))&set(acn))
RE_cells = list(set(Get_Tuned_Cells(day_folder, 'RE'))&set(acn))
No_tune_cells = lt.List_Subtraction(acn, LE_cells+RE_cells)
LE_cells.sort()
RE_cells.sort()
No_tune_cells.sort()
new_list = LE_cells+RE_cells+No_tune_cells
in_out_range = cell_in_mask+cell_out_mask
a = correlation_matrix.copy()
b = a.reindex(columns = new_list,index = new_list)
Single_Mask_Visualize(all_cell_dic,No_tune_cells)


#%%  Do this on regressed data.
correlation_matrix_reg = pd.DataFrame(index = acn,columns = acn)
for i,cell_A in tqdm(enumerate(acn)):
    for j,cell_B in enumerate(acn):
        c_r,_ = stats.pearsonr(regressed_PCA.loc[cell_A,:],regressed_PCA.loc[cell_B,:])
        correlation_matrix_reg.loc[cell_A,cell_B] = c_r
correlation_matrix_reg = correlation_matrix_reg.fillna(0)

reindex_corr_reg = correlation_matrix_reg.reindex(columns = new_list,index = new_list)
in_out_reindex = correlation_matrix_reg.reindex(columns = in_out_range,index = in_out_range)
#%% seperate cell type, get LE/LE,RE/RE,No_tune/No_tune disps.
LE_LE = correlation_matrix_reg.loc[LE_cells,LE_cells].values.flatten()
LE_RE = correlation_matrix_reg.loc[LE_cells,RE_cells].values.flatten()
RE_RE = correlation_matrix_reg.loc[RE_cells,RE_cells].values.flatten()
No_tune_No_tune = correlation_matrix_reg.loc[No_tune_cells,No_tune_cells].values.flatten()

plt.hist(random.sample(list(LE_LE),12320),bins = 100,alpha = 0.6)
plt.hist(LE_RE,bins = 100,alpha = 0.6)
plt.hist(No_tune_No_tune,bins = 100)

in_in = correlation_matrix_reg.loc[cell_in_mask,cell_in_mask].values.flatten()
in_out = correlation_matrix_reg.loc[cell_in_mask,cell_out_mask].values.flatten()
plt.hist(in_in,bins = 100,alpha = 0.6)
plt.hist(random.sample(list(in_out),14884),bins = 100,alpha = 0.6)




