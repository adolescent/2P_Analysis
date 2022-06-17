# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 12:55:05 2022

@author: adolescent

This script will compare similarity between 
"""


import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt

#%%
day_folder = r'F:\_Data_Temp\220420_L91'
workpath = r'F:\_Data_Temp\220420_L91\_CAIMAN'
all_cell_dic = ot.Load_Variable(workpath,'All_Series_Dic.pkl')
tuning_dic = ot.Load_Variable(workpath,'Cell_Tuning_Dic.pkl')
cells_in_tune = ot.Load_Variable(workpath,'Tuning_Property.pkl')
acn = list(all_cell_dic.keys())
#%% Test how fit different from each other.
fit_diff = []
for i,cc in enumerate(acn):
    if (tuning_dic[cc]['Fitted_Orien'] != 'No_Tuning') and (tuning_dic[cc]['Orien_Preference'] != 'No_Tuning'):
        c_diff = abs(tuning_dic[cc]['Fitted_Orien']-float(tuning_dic[cc]['Orien_Preference'][5:]))
        fit_diff.append(min(c_diff,180-c_diff))
    
plt.hist(fit_diff,bins = 25)
#%% Compare PC with stim graphs.

