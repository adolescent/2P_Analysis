'''
Heat map keeps the same as ver 5.
Add spontaneous annotated version here.

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
import umap
import umap.plot
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
from Cell_Class.UMAP_Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *



wp = r'D:\_All_Spon_Data_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(wp,'Cell_Class.pkl')
spon_series = ot.Load_Variable(wp,'Spon_Before.pkl').reset_index(drop = True)

import warnings
warnings.filterwarnings("ignore")


#%%###################### FIG 1C-A HEATMAPS ##############################
# get spon,stim,shuffle frames.
orien_series = ac.Z_Frames['1-007']
spon_shuffle = Spon_Shuffler(spon_series,method='phase')
spon_shuffle_frame = pd.DataFrame(spon_shuffle,columns = spon_series.columns,index = spon_series.index)

#%% Sort Orien By Cells actually we sort only by raw data.
rank_index = pd.DataFrame(index = ac.acn,columns=['Best_Orien','Sort_Index','Sort_Index2'])
for i,cc in enumerate(ac.acn):
    rank_index.loc[cc]['Best_Orien'] = ac.all_cell_tunings[cc]['Best_Orien']
    if ac.all_cell_tunings[cc]['Best_Orien'] == 'False':
        rank_index.loc[cc]['Sort_Index']=-1
        rank_index.loc[cc]['Sort_Index2']=0
    else:
        orien_tunings = float(ac.all_cell_tunings[cc]['Best_Orien'][5:])
        # rank_index.loc[cc]['Sort_Index'] = np.sin(np.deg2rad(orien_tunings))
        rank_index.loc[cc]['Sort_Index'] = orien_tunings
        rank_index.loc[cc]['Sort_Index2'] = np.cos(np.deg2rad(orien_tunings))
# actually we sort only by raw data.
sorted_cell_sequence = rank_index.sort_values(by=['Sort_Index'],ascending=False)
# and we try to reindex data.
sorted_stim_response = orien_series.T.reindex(sorted_cell_sequence.index).T
sorted_spon_response = spon_series.T.reindex(sorted_cell_sequence.index).T
sorted_shuffle_response = spon_shuffle_frame.T.reindex(sorted_cell_sequence.index).T

#%% Plot Cell Stim Maps
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10,8),dpi = 180)
cbar_ax = fig.add_axes([1, .35, .01, .3])
label_size = 14
title_size = 18

vmax = 4
vmin = -2
sns.heatmap((sorted_stim_response .iloc[1000:1650,:].T),center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax)
sns.heatmap(sorted_spon_response.iloc[4700:5350,:].T,center = 0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax)
sns.heatmap(sorted_shuffle_response.iloc[4700:5350,:].T,center = 0,xticklabels=False,yticklabels=False,ax = axes[2],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax,cbar_kws={'label': 'Z Scored Activity'})
cbar_ax.yaxis.label.set_size(label_size)

xticks = np.array([0,100,200,300,400,500])
axes[2].set_xticks(xticks)
axes[2].set_xticklabels([0,100,200,300,400,500])
from matplotlib.patches import Rectangle
axes[0].add_patch(Rectangle((175,0), 6, 520, fill=False, edgecolor='yellow', lw=1,alpha = 0.8))
axes[1].add_patch(Rectangle((461,0), 6, 520, fill=False, edgecolor='yellow', lw=1,alpha = 0.8))
# axes[1].add_patch(Rectangle((536,0), 6, 520, fill=False, edgecolor='red', lw=1,alpha = 0.8))
axes[2].add_patch(Rectangle((461,0), 6, 520, fill=False, edgecolor='yellow', lw=1,alpha = 0.8))

axes[0].set_title('Stim-induced Response',size = title_size)
axes[1].set_title('Spontaneous Response',size = title_size)
axes[2].set_title('Shuffled Spontaneous Response',size = title_size)

axes[2].set_xlabel(f'Time (s)',size = label_size)
axes[2].set_ylabel(f'Cells',size = label_size)
axes[1].set_ylabel(f'Cells',size = label_size)
axes[0].set_ylabel(f'Cells',size = label_size)

# annotate cell number on it.
for i in range(3):
    # axes[i].set_yticks([0,100,200,300,400,500])
    axes[i].set_yticks([0,180,360,524])
    # axes[i].set_yticklabels([0,100,200,300,400,500],rotation = 90,fontsize = 7)
    axes[i].set_yticklabels([0,180,360,524],rotation = 90,fontsize = 8)

fps = 1.301
axes[2].set_xticks([0*fps,100*fps,200*fps,300*fps,400*fps,500*fps])
axes[2].set_xticklabels([0,100,200,300,400,500],fontsize = 8)

fig.tight_layout()
plt.show()

#%% ####################### Fig 1D Ver 2, Annotate Stimulus onto Graphs. ##############
