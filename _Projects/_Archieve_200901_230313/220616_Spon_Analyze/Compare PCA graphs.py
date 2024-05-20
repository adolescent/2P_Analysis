# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 12:55:05 2022

@author: adolescent

This script will compare similarity between 
"""


import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt

#%% read in 3 data sheets.
day_folder_91 = r'F:\_Data_Temp\220420_L91'
workpath_91 = r'F:\_Data_Temp\220420_L91\_CAIMAN'
all_cell_dic_91 = ot.Load_Variable(workpath_91,'All_Series_Dic.pkl')
tuning_dic_91 = ot.Load_Variable(workpath_91,'Cell_Tuning_Dic.pkl')
cells_in_tune_91 = ot.Load_Variable(workpath_91,'Tuning_Property.pkl')
acn_91 = list(all_cell_dic_91.keys())

day_folder_85 = r'F:\_Data_Temp\220407_L85'
workpath_85 = r'F:\_Data_Temp\220407_L85\_CAIMAN'
all_cell_dic_85 = ot.Load_Variable(workpath_85,'All_Series_Dic.pkl')
tuning_dic_85 = ot.Load_Variable(workpath_85,'Cell_Tuning_Dic.pkl')
cells_in_tune_85 = ot.Load_Variable(workpath_85,'Tuning_Property.pkl')
acn_85 = list(all_cell_dic_85.keys())


day_folder_76 = r'F:\_Data_Temp\210831_L76'
workpath_76 = r'F:\_Data_Temp\210831_L76\_CAIMAN'
all_cell_dic_76 = ot.Load_Variable(workpath_76,'All_Series_Dic.pkl')
tuning_dic_76 = ot.Load_Variable(workpath_76,'Cell_Tuning_Dic.pkl')
cells_in_tune_76 = ot.Load_Variable(workpath_76,'Tuning_Property.pkl')
acn_76 = list(all_cell_dic_76.keys())



#%% Test how fit different from each other.
fit_diff = []

for i,cc in enumerate(acn_85):
    if (tuning_dic_85[cc]['Fitted_Orien'] != 'No_Tuning') and (tuning_dic_85[cc]['Orien_Preference'] != 'No_Tuning'):
        c_diff = abs(tuning_dic_85[cc]['Fitted_Orien']-float(tuning_dic_85[cc]['Orien_Preference'][5:]))
        fit_diff.append(min(c_diff,180-c_diff))
for i,cc in enumerate(acn_91):
    if (tuning_dic_91[cc]['Fitted_Orien'] != 'No_Tuning') and (tuning_dic_91[cc]['Orien_Preference'] != 'No_Tuning'):
        c_diff = abs(tuning_dic_91[cc]['Fitted_Orien']-float(tuning_dic_91[cc]['Orien_Preference'][5:]))
        fit_diff.append(min(c_diff,180-c_diff))
for i,cc in enumerate(acn_76):
    if (tuning_dic_76[cc]['Fitted_Orien'] != 'No_Tuning') and (tuning_dic_76[cc]['Orien_Preference'] != 'No_Tuning'):
        c_diff = abs(tuning_dic_76[cc]['Fitted_Orien']-float(tuning_dic_76[cc]['Orien_Preference'][5:]))
        fit_diff.append(min(c_diff,180-c_diff))

plt.hist(fit_diff,bins = 25)
#%% Get all cell by cell correlation matrix.




