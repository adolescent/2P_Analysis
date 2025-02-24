# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:44:53 2022

@author: ZR


This part generate the map of distance-tuing-corr plots.

"""

import OS_Tools_Kit as ot
import Graph_Operation_Kit as gt
from Series_Analyzer.Preprocessor_Cai import Pre_Processor_Cai
import matplotlib.pyplot as plt
from Series_Analyzer.Pairwise_Correlation import Pairwise_Corr_Core
import seaborn as sns
import statsmodels.api as sm
import pandas as pd
import numpy as np

day76 = r'D:\ZR\_Temp_Data\220630_L76_2P'
day85 = r'D:\ZR\_Temp_Data\220706_L85_LM'
day91 = r'D:\ZR\_Temp_Data\220420_L91'
wp91 = r'D:\ZR\_Temp_Data\220420_L91\_CAIMAN'
wp85 = r'D:\ZR\_Temp_Data\220706_L85_LM\_CAIMAN'
wp76 = r'D:\ZR\_Temp_Data\220630_L76_2P\_CAIMAN'

#%% 91 first.
Frame_1_91 = Pre_Processor_Cai(day91,'Run001',start_frame = 3000)
acd91 = ot.Load_Variable(wp91,'All_Series_Dic.pkl')
acn91 = list(acd91.keys())
tuningdic91 = ot.Load_Variable(wp91,'Cell_Tuning_Dic.pkl')
pc91_whole = Pairwise_Corr_Core(acd91, tuningdic91, Frame_1_91)
sns.scatterplot(data = pc91_whole,x = 'Dist',y = 'Corr',s = 2)
ot.Save_Variable(wp91, 'Pair_Corr_Whole_1', pc91_whole)
#%% 76
Frame_1_76 = Pre_Processor_Cai(day76,'Run001',start_frame = 4000)
acd76 = ot.Load_Variable(wp76,'All_Series_Dic.pkl')
acn76 = list(acd76.keys())
tuningdic76 = ot.Load_Variable(wp76,'Cell_Tuning_Dic.pkl')
pc76_whole = Pairwise_Corr_Core(acd76, tuningdic76, Frame_1_76)
sns.scatterplot(data = pc76_whole,x = 'Dist',y = 'Corr',s = 2)
ot.Save_Variable(wp76, 'Pair_Corr_Whole_1', pc76_whole)
#%% 85
Frame_1_85 = Pre_Processor_Cai(day85,'Run001',start_frame = 4000)
acd85 = ot.Load_Variable(wp85,'All_Series_Dic.pkl')
acn85 = list(acd85.keys())
tuningdic85 = ot.Load_Variable(wp85,'Cell_Tuning_Dic.pkl')
pc85_whole = Pairwise_Corr_Core(acd85, tuningdic85, Frame_1_85)
sns.scatterplot(data = pc85_whole,x = 'Dist',y = 'Corr',s = 2)
ot.Save_Variable(wp85, 'Pair_Corr_Whole_1', pc85_whole)
#%% in total,pooling
temp_workfolder = r'D:\ZR\_Temp_Data\220711_temp'
# generate pooling data frame.
pc76_whole['Animal'] = '76'
pc85_whole['Animal'] = '85'
pc91_whole['Animal'] = '91'
pooled_pc = pd.concat([pc91_whole,pc76_whole,pc85_whole])
ot.Save_Variable(temp_workfolder, 'Paircorr_Run01_pool', pooled_pc)
# example of get subset:
    #pc91_returned = pooled_pc.loc[pooled_pc['Animal']== '91',:]
# scatter plot.
sns.scatterplot(data = pooled_pc,x = 'Dist',y = 'Corr',s = 2,hue = 'Animal',hue_order = ['91','76','85'])
# or lm plot.
sns.lmplot(data = pooled_pc,x = 'Dist',y = 'Corr',scatter_kws = {'s':2,'alpha':0.15},hue = 'Animal')
#%% Linear fit model to get parameter.
dist91 = pc91_whole['Dist']
Y = pc91_whole['Corr']
X = sm.add_constant(dist91)
model91 = sm.OLS(Y,X)
result91 = model91.fit()
result91.summary()

dist85 = pc85_whole['Dist']
Y = pc85_whole['Corr']
X = sm.add_constant(dist85)
model85 = sm.OLS(Y,X)
result85 = model85.fit()
result85.summary()

dist76 = pc76_whole['Dist']
Y = pc76_whole['Corr']
X = sm.add_constant(dist76)
model76 = sm.OLS(Y,X)
result76 = model76.fit()
result76.summary()
#%% Non linear fit.
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
def Reci_Func(dist,const,slope,bias):
    corr = const+slope*(1/(dist+bias))
    return corr
# fit76
para_76, covar_76 = curve_fit(Reci_Func,xdata = pc76_whole['Dist'],ydata = pc76_whole['Corr'])
y_pred_76 = Reci_Func(pc76_whole['Dist'], *para_76)
r2_76 = r2_score(pc76_whole['Corr'], y_pred_76)
dist_range = np.array(range(1,620))
plt.scatter(x =  pc76_whole['Dist'],y =  pc76_whole['Corr'],s = 1,alpha = 0.5)
plt.plot(dist_range,Reci_Func(dist_range,*para_76),color = 'r')
#fit 85
para_85, covar_85 = curve_fit(Reci_Func,xdata = pc85_whole['Dist'],ydata = pc85_whole['Corr'])
y_pred_85 = Reci_Func(pc85_whole['Dist'], *para_85)
r2_85 = r2_score(pc85_whole['Corr'], y_pred_85)
dist_range = np.array(range(1,620))
plt.scatter(x =  pc85_whole['Dist'],y =  pc85_whole['Corr'],s = 1,alpha = 0.5)
plt.plot(dist_range,Reci_Func(dist_range,*para_85),color = 'r')
#fit 91
para_91, covar_91 = curve_fit(Reci_Func,xdata = pc91_whole['Dist'],ydata = pc91_whole['Corr'])
y_pred_91 = Reci_Func(pc91_whole['Dist'], *para_91)
r2_91 = r2_score(pc91_whole['Corr'], y_pred_91)
dist_range = np.array(range(1,620))
plt.scatter(x =  pc91_whole['Dist'],y =  pc91_whole['Corr'],s = 1,alpha = 0.5)
plt.plot(dist_range,Reci_Func(dist_range,*para_91),color = 'r')
#%% Regress correlation on non-linear pattern.
regressed_pc91_whole = pc91_whole.copy()
regressed_pc91_whole['Corr'] = pc91_whole['Corr']-Reci_Func(pc91_whole['Dist'], *para_91)+para_91[0]
plt.scatter(x =  regressed_pc91_whole['Dist'],y =  regressed_pc91_whole['Corr'],s = 1,alpha = 0.5)

regressed_pc85_whole = pc85_whole.copy()
regressed_pc85_whole['Corr'] = pc85_whole['Corr']-Reci_Func(pc85_whole['Dist'], *para_85)+para_85[0]
plt.scatter(x =  regressed_pc85_whole['Dist'],y =  regressed_pc85_whole['Corr'],s = 1,alpha = 0.5)

regressed_pc76_whole = pc76_whole.copy()
regressed_pc76_whole['Corr'] = pc76_whole['Corr']-Reci_Func(pc76_whole['Dist'], *para_76)+para_76[0]
plt.scatter(x =  regressed_pc76_whole['Dist'],y =  regressed_pc76_whole['Corr'],s = 1,alpha = 0.5)

# Test after regression
dist_t = regressed_pc76_whole['Dist']
Y = regressed_pc76_whole['Corr']
X = sm.add_constant(dist_t)
model_t = sm.OLS(Y,X)
result_t = model_t.fit()
result_t.summary()
sns.lmplot(data = regressed_pc76_whole,x = 'Dist',y = 'Corr',scatter_kws = {'s':2,'alpha':0.15})
# generate and save all 
pooled_pc_reg = pd.concat([regressed_pc91_whole,regressed_pc76_whole,regressed_pc85_whole])
ot.Save_Variable(temp_workfolder, 'Paircorr_Run01_pool_Dist_Regressed', pooled_pc_reg)
#%%


