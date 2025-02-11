'''
This will try to analysis the cell dff's relationship with tuning value.
Negative relationship will be shown.
'''

#%%
from Cell_Class.Stim_Calculators import Stim_Cells
from Cell_Class.Format_Cell import Cell
import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import cv2
from Kill_Cache import kill_all_cache
from sklearn.model_selection import cross_val_score
from sklearn import svm
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
from Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *
import warnings
warnings.filterwarnings("ignore")

wp = r'D:\_All_Spon_Data_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(wp,'Cell_Class.pkl')
spon_series = ot.Load_Variable(wp,'Spon_Before.pkl').reset_index(drop = True)
save_path = r'D:\_GoogleDrive_Files\#Figs\240627_Figs_FF1\Fig1'
all_path_dic = list(ot.Get_Subfolders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)


#%%
'''
Step1, we calculate example loc's dff info, and tuning info.
'''
exp_tuning = ac.all_cell_tunings
spon_dff = ac.Get_dFF_Frames('1-001',0.1,8500,13852)
dff_mean = spon_dff.mean(0)
dff_best = np.zeros(spon_dff.shape[1])
for i in range(spon_dff.shape[1]):
    c_train = spon_dff[:,i]
    best_id = np.argpartition(c_train,-535)[-535:]
    best_resp=c_train[best_id].mean()
    dff_best[i] = best_resp
# generate plotable matrix.
plotable = pd.DataFrame(0.0,index = range(spon_dff.shape[1]),columns = ['OD_Index','Orien_Index','dff_mean','dff_best'])
plotable['OD_Index']= np.array(exp_tuning.loc['OD_index'])
plotable['Orien_Index']= np.array(exp_tuning.loc['Orien_index'])
plotable['dff_mean'] = dff_mean
plotable['dff_best'] = dff_best

#%% Get all location's matrix.

for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    spon_series = ot.Load_Variable(cloc,'Spon_Before.pkl')
    spon_dff = ac.Get_dFF_Frames('1-001',0.1,spon_series.index[0],spon_series.index[-1])
    exp_tuning = ac.all_cell_tunings
    exp_hv = ac.Orien_t_graphs['H-V'].loc['CohenD']
    exp_ao = ac.Orien_t_graphs['A-O'].loc['CohenD']
    dff_mean = spon_dff.mean(0)
    dff_best = np.zeros(spon_dff.shape[1])
    for j in range(spon_dff.shape[1]):
        c_train = spon_dff[:,j]
        base_num = int(len(spon_series)*0.1)
        best_id = np.argpartition(c_train,-base_num)[-base_num:]
        best_resp=c_train[best_id].mean()
        dff_best[j] = best_resp
    plotable = pd.DataFrame(0.0,index = range(spon_dff.shape[1]),columns = ['OD_Index','Orien_Index','dff_mean','dff_best'])
    plotable['OD_Index']= abs(np.array(exp_tuning.loc['OD']))
    c_best_orien = np.array([abs(exp_hv),abs(exp_ao)]).max(0)
    # plotable['Orien_Index']= np.array(exp_tuning.loc['Orien_index'])
    plotable['Orien_Index']= c_best_orien
    plotable['dff_mean'] = dff_mean
    plotable['dff_best'] = dff_best
    if i ==0:
        all_plotable = copy.deepcopy(plotable)
    else:
        all_plotable = pd.concat([all_plotable,plotable])
#%% Plot part
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (5,4),dpi = 300)
sns.scatterplot(data = all_plotable,x = 'dff_best',y = 'Orien_Index',s = 2,ax = ax,linewidth = 0)
# ax.set_ylim(-1,4)
stats.pearsonr(plotable['dff_best'],plotable['Orien_Index'])
# no significant corr shown.