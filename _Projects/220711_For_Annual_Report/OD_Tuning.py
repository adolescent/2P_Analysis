# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:56:18 2022

@author: ZR

This part is used to deal with OD/ Corr method.

"""

#%% First, whole series corr.
import My_Wheels.OS_Tools_Kit as ot
import matplotlib.pyplot as plt
import seaborn as sns


wp = r'F:\_Data_Temp\220711_temp'


pc_raw = ot.Load_Variable(wp,r'Paircorr_Run01_pool.pkl')
pc_raw = pc_raw.reset_index(drop = True)
pc_regressed = ot.Load_Variable(wp,r'Paircorr_Run01_pool_Dist_Regressed.pkl')
pc_regressed = pc_regressed.reset_index(drop = True)



#%% get OD-corr graph.
# 2 params, abs(OD_index sum) and abs(OD_index diff).
od_diff = abs(pc_raw['OD_A']-pc_raw['OD_B'])
od_sum = abs(pc_raw['OD_A']+pc_raw['OD_B'])
pc_raw['OD_diff'] = od_diff
pc_raw['OD_sum'] = od_sum
pc91 = pc_raw.loc[pc_raw['Animal']== '91',:]
pc85 = pc_raw.loc[pc_raw['Animal']== '85',:]
pc76 = pc_raw.loc[pc_raw['Animal']== '76',:]

sns.scatterplot(data = pc_raw, x = 'OD_diff',y = 'Corr',s = 2,hue = 'Animal')
sns.lmplot(data = pc_raw, x = 'OD_diff',y = 'Corr',scatter_kws = {'s':2,'alpha':0.15},hue = 'Animal')
# and kde scatter plot.
f, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(data = pc76, x = 'OD_diff',y = 'Corr',fill=True,thresh=0, levels=100, cmap="rocket", color="w", linewidths=1)
# Only diff regression.

# both diff and sum regression
    # sum vif vs diff,how independent they are.
    # fit vs predict-sse graph

#%% work on distance regressed data.

od_diff_reg = abs(pc_raw['OD_A']-pc_raw['OD_B'])
od_sum_reg = abs(pc_raw['OD_A']+pc_raw['OD_B'])
pc_regressed['OD_diff'] = od_diff_reg
pc_regressed['OD_sum'] = od_sum_reg

sns.scatterplot(data = pc_regressed, x = 'OD_diff',y = 'Corr',s = 2,hue = 'Animal')
sns.lmplot(data = pc_regressed, x = 'OD_diff',y = 'Corr',scatter_kws = {'s':2,'alpha':0.15},hue = 'Animal')




