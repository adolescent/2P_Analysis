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
import numpy as np
import statsmodels.api as sm
from Series_Analyzer.VIF_Checker import VIF_Check
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


wp = r'D:\ZR\_Temp_Data\220711_temp'


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


#%%  Only diff regression.
X = pc85['OD_diff']
Y = pc85['Corr']
X = sm.add_constant(X)
model85 = sm.OLS(Y,X)
result85 = model85.fit()
result85.summary()
X_fit = np.arange(-0.1,2.1,0.01)
Y_fit = result85.params[0]+result85.params[1]*X_fit
# and kde scatter plot.
f, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(data = pc85, x = 'OD_diff',y = 'Corr',fill=True,thresh=0, levels=100, cmap="rocket", color="w", linewidths=1)
sns.lineplot(X_fit,Y_fit,color = 'y')




#%% both diff and sum regression

# sum vif vs diff,how independent they are.
vif76 = VIF_Check(pc76, checked_column_name=['Dist','OD_diff','OD_sum'])
vif85 = VIF_Check(pc85, checked_column_name=['Dist','OD_diff','OD_sum'])
vif91 = VIF_Check(pc91, checked_column_name=['Dist','OD_diff','OD_sum'])

# 2-variance regression
def Dist_OD_Model(X,const,bias,A,B):
    dist,OD_diff = X
    Corr = const+A/(dist+bias)+B*OD_diff
    return Corr

para_76, covar_76 = curve_fit(Dist_OD_Model,xdata = [pc76['Dist'],pc76['OD_diff']],ydata = pc76['Corr'])

y_pred_76 = Dist_OD_Model([pc76['Dist'],pc76['OD_diff']], *para_76)
r2_76 = r2_score(pc76['Corr'], y_pred_76)

# fit vs predict-sse graph
    
    
    

#%% This part will produce orientation vs OD.

# total VIF, is each variable independent?



