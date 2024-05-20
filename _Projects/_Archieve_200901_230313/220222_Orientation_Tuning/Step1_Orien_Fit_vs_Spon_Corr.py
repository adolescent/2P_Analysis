# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:12:51 2022

@author: ZR

This part use spontaneous before to test orientation/ 

"""

from Stimulus_Cell_Processor.Orien_Fit import Orientation_Pref_Fit
import OS_Tools_Kit as ot
import cv2
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
from Series_Analyzer.Cell_Frame_PCA import Do_PCA,PCA_Regression
import matplotlib.pyplot as plt
from Series_Analyzer.Single_Component_Visualize import Single_Mask_Visualize
from Stimulus_Cell_Processor.Get_Tuning import Get_Tuned_Cells
import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns


#%%
day_folder = r'G:\Test_Data\2P\210831_L76_2P'
Oft = Orientation_Pref_Fit(day_folder,orien_run = 'Run002')
fitted_orien = Oft.One_Key_Fit()
Run01_Frame = Pre_Processor(day_folder,'Run001',start_time = 6000)
#%% Analyze cross correlation vs orien difference..
acn = list(Run01_Frame.index)
Cross_Corr = ot.Load_Variable(day_folder,'Corr_Matrix.pkl')
New_Cross_Corr = pd.DataFrame()
total_pair_num = len(Cross_Corr)

for i in tqdm(range(total_pair_num)):
    c_pair = Cross_Corr.iloc[i,:]
    c_A_cell = c_pair['Cell_A']
    c_B_cell = c_pair['Cell_B']
    c_A_angle = fitted_orien[c_A_cell]['Best_Angle']
    c_B_angle = fitted_orien[c_B_cell]['Best_Angle']
    raw_angle_diff = abs(c_A_angle-c_B_angle)%180
    angle_diff = min(raw_angle_diff,180-raw_angle_diff)
    c_pair['Angle_Diff'] = angle_diff
    New_Cross_Corr[i] = c_pair
#%% Plot orien_diff vs corr.
New_Cross_Corr = New_Cross_Corr.T
New_Cross_Corr['Pearsonr'] = New_Cross_Corr['Pearsonr'].astype('f8')
New_Cross_Corr['Angle_Diff'] = New_Cross_Corr['Angle_Diff'].astype('f8')
ot.Save_Variable(day_folder, 'Corr_Matrix_Run01', New_Cross_Corr)
sns.scatterplot(data = New_Cross_Corr,y = 'Pearsonr',x = 'Angle_Diff',s = 3)
sns.regplot(data = New_Cross_Corr,y = 'Pearsonr',x = 'Angle_Diff',scatter_kws = {'s' : 1.5})    
    
from sklearn.linear_model import LinearRegression
X = New_Cross_Corr.loc[:,['Angle_Diff']]
Y = New_Cross_Corr.loc[:,['Pearsonr']]
model = LinearRegression()
model.fit(X,Y)
model.score(X,Y)
import statsmodels.api as sm
X2 = sm.add_constant(X)
est = sm.OLS(Y,X2)
est2 = est.fit()
est2.summary()


#%% Get group dist and group angle diff.
New_Cross_Corr['grouped_dist'] = (New_Cross_Corr['Dist']//75+1)*75
New_Cross_Corr['grouped_angle_diff'] = (New_Cross_Corr['Angle_Diff']//1)*1

Angle_Dist_Corr = pd.DataFrame(columns = ['Corr','Angle','Dist'])
data_grouped = dict(list(New_Cross_Corr.groupby('grouped_dist')))
counter = 0
all_dist = list(data_grouped.keys())
for i,c_dist in tqdm(enumerate(all_dist)):
    c_dist_group = data_grouped[c_dist]
    c_angle_group = dict(list(c_dist_group.groupby('grouped_angle_diff')))
    c_angles = list(c_angle_group.keys())
    for j,c_angle in enumerate(c_angles):
        avr_corr = c_angle_group[c_angle]['Pearsonr'].mean()
        Angle_Dist_Corr.loc[counter] = [avr_corr,c_angle,c_dist]
        counter+=1

dist_angle_corr = Angle_Dist_Corr.pivot(index = 'Dist',columns = 'Angle',values = 'Corr')    
sns.heatmap(dist_angle_corr,vmax = 0.4,vmin = 0.25)
#%% get total average graphs.
New_Cross_Corr['grouped_angle_diff'] = (New_Cross_Corr['Angle_Diff']//1)*1
sns.lineplot(data = New_Cross_Corr,x = 'grouped_angle_diff',y = 'Pearsonr')
New_Cross_Corr['gropuped_OD_diff'] = (New_Cross_Corr['OD_Tuning_diff']//0.02)*0.02
sns.lineplot(data = New_Cross_Corr,x = 'gropuped_OD_diff',y = 'Pearsonr')

#%% Get OD indexed data.
all_tunings = ot.Load_Variable(day_folder,r'All_Tuning_Property.tuning')
pair_num = len(New_Cross_Corr)
New_Cross_Corr['OD_index_A'] = 0
New_Cross_Corr['OD_index_B'] = 0
New_Cross_Corr['OD_index_diff'] = 0
for i in tqdm(range(pair_num)):
    c_pair = New_Cross_Corr.iloc[i,:]
    c_A_cell = c_pair['Cell_A']
    c_B_cell = c_pair['Cell_B']
    A_OD_index = np.clip(all_tunings[c_A_cell]['LE']['Tuning_Index'],-1,1)# LE as positive,RE as negative
    B_OD_index = np.clip(all_tunings[c_B_cell]['LE']['Tuning_Index'],-1,1)
    OD_index_diff = abs(A_OD_index-B_OD_index)
    New_Cross_Corr.loc[i,'OD_index_A'] = A_OD_index
    New_Cross_Corr.loc[i,'OD_index_B'] = B_OD_index
    New_Cross_Corr.loc[i,'OD_index_diff'] = OD_index_diff
New_Cross_Corr['Grouped_OD_diff'] = (New_Cross_Corr['OD_index_diff']//0.05)*0.05

sns.scatterplot(data = New_Cross_Corr,x = 'OD_index_diff',y = 'Pearsonr',s = 1.5)
sns.lmplot(data = New_Cross_Corr,x = 'OD_index_diff',y = 'Pearsonr',scatter_kws = {'s' : 1})
sns.lmplot(data = New_Cross_Corr,x = 'Grouped_OD_diff',y = 'Pearsonr',scatter_kws = {'s' : 0.5})
sns.lineplot(data = New_Cross_Corr,x = 'Grouped_OD_diff',y = 'Pearsonr')
# linear model
X = New_Cross_Corr['OD_index_diff']
Y = New_Cross_Corr['Pearsonr']
X2 = sm.add_constant(X)
est = sm.OLS(Y,X2)
est2 = est.fit()
est2.summary()
#%% Tag OD Diff in 4 parts.
tune_thres = 0.25
New_Cross_Corr['OD_diff_tag'] = None
for i in tqdm(range(total_pair_num)):
    c_pair = New_Cross_Corr.iloc[i,:]
    A_OD_index = c_pair['OD_index_A']
    B_OD_index = c_pair['OD_index_B']
    if abs(A_OD_index)>tune_thres and abs(B_OD_index)>tune_thres and A_OD_index*B_OD_index>0:
        New_Cross_Corr.loc[i,'OD_diff_tag'] = 'Same'
    elif abs(A_OD_index)>tune_thres and abs(B_OD_index)>tune_thres and A_OD_index*B_OD_index<0:
        New_Cross_Corr.loc[i,'OD_diff_tag'] = 'Different'
    elif abs(A_OD_index)<tune_thres and abs(B_OD_index)<tune_thres:
        New_Cross_Corr.loc[i,'OD_diff_tag'] = 'All_None'
    else:
        New_Cross_Corr.loc[i,'OD_diff_tag'] = 'Single_None'
    
Grouped_OD_tune_Corr = dict(list(New_Cross_Corr.groupby('OD_diff_tag')))    

sns.lmplot(data = New_Cross_Corr,x = 'OD_index_diff',y = 'Pearsonr',scatter_kws = {'s' : 0.5},hue = 'OD_diff_tag')
sns.lmplot(data = Grouped_OD_tune_Corr['Single_None'],x = 'OD_index_diff',y = 'Pearsonr',scatter_kws = {'s' : 0.5},hue = 'OD_diff_tag')

sns.lineplot(data = New_Cross_Corr,x = 'Grouped_OD_diff',y = 'Pearsonr',hue = 'OD_diff_tag')
#%% linear model
# Add average OD tuning as a para.
New_Cross_Corr['OD_Tuning_avr'] = ((New_Cross_Corr['OD_index_A']+New_Cross_Corr['OD_index_B'])/2)
X = New_Cross_Corr[['OD_index_diff','OD_Tuning_avr']]
Y = New_Cross_Corr['Pearsonr']
X2 = sm.add_constant(X)
est = sm.OLS(Y,X2)
est2 = est.fit()
est2.summary()






