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
import statsmodels.stats.api as sms
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
X = pc_raw['OD_diff']
Y = pc_raw['Corr']
X = sm.add_constant(X)
model = sm.OLS(Y,X)
result = model.fit()
result.summary()
X_fit = np.arange(-0.1,2.1,0.01)
Y_fit = result.params[0]+result.params[1]*X_fit
# and kde scatter plot.
f, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(data = pc_raw, x = 'OD_diff',y = 'Corr',fill=True,thresh=0, levels=100, cmap="rocket", color="w", linewidths=1)
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

para_85, covar_85 = curve_fit(Dist_OD_Model,xdata = [pc85['Dist'],pc85['OD_diff']],ydata = pc85['Corr'])

y_pred_85 = Dist_OD_Model([pc85['Dist'],pc85['OD_diff']], *para_85)
r2_85 = r2_score(pc85['Corr'], y_pred_85)
# Not very good, just ignore here.


#%% This part will produce orientation vs Corr.
# Ignore untuned cell.
pc_tuned = pc_raw.copy()
pc_tuned = pc_tuned[pc_tuned.OD_A != -999]
pc_tuned = pc_tuned[pc_tuned.OD_B != -999]
pc_tuned = pc_tuned[pc_tuned.Orien_A != -999]
pc_tuned = pc_tuned[pc_tuned.Orien_B != -999]


raw_orien_diff = abs(pc_tuned['Orien_A']-pc_tuned['Orien_B'])
raw_orien_diff[raw_orien_diff>90]=(180-raw_orien_diff)
pc_tuned['Orien_diff'] = raw_orien_diff

pc76_t = pc_tuned.loc[pc_tuned['Animal']== '76',:]
pc85_t = pc_tuned.loc[pc_tuned['Animal']== '85',:]
pc91_t = pc_tuned.loc[pc_tuned['Animal']== '91',:]
sns.scatterplot(data = pc_tuned, x = 'Orien_diff',y = 'Corr',s = 2,hue = 'Animal')
sns.lmplot(data = pc_tuned, x = 'Orien_diff',y = 'Corr',scatter_kws = {'s':2,'alpha':0.15},hue = 'Animal')

#%% fit orien diff and corr.
X = pc_tuned['Orien_diff']
Y = pc_tuned['Corr']
X = sm.add_constant(X)
model = sm.OLS(Y,X)
result = model.fit()
result.summary()
X_fit = np.arange(-5,98,0.5)
Y_fit = result.params[0]+result.params[1]*X_fit
# and kde scatter plot.
f, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(data = pc_tuned, x = 'Orien_diff',y = 'Corr',fill=True,thresh=0, levels=100, cmap="rocket", color="w", linewidths=1)
sns.lineplot(X_fit,Y_fit,color = 'y')




#%% This part will generate a global fit model, un tuned cell need to be added into here.

# How independent tunings are.
vif76 = VIF_Check(pc76_t, checked_column_name=['Dist','OD_diff','Orien_diff'])
vif85 = VIF_Check(pc85_t, checked_column_name=['Dist','OD_diff','Orien_diff'])
vif91 = VIF_Check(pc91_t, checked_column_name=['Dist','OD_diff','Orien_diff'])
vif_all = VIF_Check(pc_tuned, checked_column_name=['Dist','OD_diff','Orien_diff'])

# fit model.
X = pc85_t[['OD_diff','Orien_diff']]
Y = pc85_t['Corr']
X = sm.add_constant(X)
model = sm.OLS(Y,X)
result = model.fit()
result.summary()
Y_pred = result.predict(X)
residue = Y-Y_pred
plt.scatter(x = Y_pred,y = residue,s = 1)
plt.hist(residue,bins = 150)
sns.kdeplot(x = Y_pred,y = residue,fill=True,thresh=0, levels=100, cmap="rocket", color="w", linewidths=1,square = True)





