'''
Here we explore the relationship with after effect strength and cell's correlation toward other cells.
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

all_path_dic = list(ot.Get_Subfolders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(3) # Run 03 bug
all_path_dic.pop(3) # OD bug
all_path_dic.pop(5) # OD bug

save_path = r'D:\_Path_For_Figs\240806_Spon_After'
all_expt_frames = ot.Load_Variable(save_path,'All_Example.pkl')

#%%
'''
Here we calculate correlation of each cell between other cells
'''
# after_len = 25
all_corr_mats = {}
for i,cloc in enumerate(all_path_dic):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable(cloc,'Cell_Class.pkl')
    spon_before = ot.Load_Variable(cloc,'Spon_Before.pkl')
    # cloc_after_effect = all_expt_frames[cloc_name][1].iloc[:after_len,:]
    # after_strength = cloc_after_effect.mean(0)
    # then we calculate each cell's correlation distribution.
    cell_num = len(ac)
    c_corr_mat = np.zeros(shape = (cell_num,cell_num-1),dtype='f8')
    for j in tqdm(range(cell_num)):
        cell_a = spon_before.loc[:,j+1]
        counter = 0
        for k in range(cell_num):
            if k != j:# avoid the same cell corr.
                cell_b = spon_before.loc[:,k+1]
                corr,_ = stats.pearsonr(cell_a,cell_b)
                c_corr_mat[j,counter] = corr
                counter+=1
    all_corr_mats[cloc_name] = c_corr_mat
ot.Save_Variable(save_path,'All_Corr_by_cell',all_corr_mats)
#%% 
'''
Then, we compare cell's correlation with after effect.
'''
after_len = 25
couping_frame = pd.DataFrame(columns = ['Loc','Cell','Avr_Corr','Std_Corr','After_effect_strength'])
for i,cloc in enumerate(all_path_dic):
    cloc_name = cloc.split('\\')[-1]
    cloc_after_effect = all_expt_frames[cloc_name][1].iloc[:after_len,:]
    after_strength = cloc_after_effect.mean(0)
    c_corr_mat = all_corr_mats[cloc_name]
    for j in range(len(after_strength)):
        cc_after = after_strength.iloc[j]
        cc_corr = c_corr_mat[j,:]
        cc_corr_mean = cc_corr.mean()
        cc_corr_std = cc_corr.std()
        couping_frame.loc[len(couping_frame),:] = [cloc_name,j+1,cc_corr_mean,cc_corr_std,cc_after]
#%% Plot coupinling with after effect.
plotable = couping_frame

plt.clf()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4),dpi = 300)
sns.scatterplot(data = plotable,y = 'Avr_Corr',x = 'After_effect_strength',s = 2,lw=0)

ax.set_ylim(-0.2,1)
ax.set_xlim(-4,4)

