'''
This part will revise graph we make from 231214_Figures. Fig 1 include all cell location & mask find and shuffled frames.

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
from Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *


work_path = r'D:\_Path_For_Figs\240123_Graph_Revised_v1\Fig1_Revised'
expt_folder = r'D:\_All_Spon_Data_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
ac.wp = expt_folder
global_avr = cv2.imread(f'{expt_folder}\Global_Average_cai.tif',0) # read as 8 bit gray scale map.
c_spon = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
ac.Regenerate_Cell_Graph()

import warnings
warnings.filterwarnings("ignore")
#%% ################################### 1. get weighted cell Mask ##################################
cell_locations = ac.new_avr_graph
# folded graph
RGB_graph = np.transpose(np.array([global_avr,global_avr,global_avr],dtype='f8'),(1,2,0))
RGB_graph = RGB_graph*0.9
RGB_graph[:,:,2] += cell_locations.astype('f8')*2
plotable_graph = RGB_graph.clip(0,255).astype('u1')
cv2.imwrite(ot.join(work_path,'Annotated_Cell_v1.png'),plotable_graph)
# to all cell graph
ac_circle = ac.Generate_Weighted_Cell(np.ones(len(ac.acn)))
cv2.imwrite(ot.join(work_path,'Cell_Circle_Map.png'),(ac_circle*255).astype('u1'))
# to all cell graph (folded)
RGB_graph = np.transpose(np.array([global_avr,global_avr,global_avr],dtype='f8'),(1,2,0))
RGB_graph = RGB_graph*0.9
RGB_graph[:,:,2] += ac_circle*255*0.35
plotable_graph = RGB_graph.clip(0,255).astype('u1')
cv2.imwrite(ot.join(work_path,'Annotated_Cell_v2.png'),plotable_graph)

#%% ################################## 2. NON SORTED REPEATS ####################################
spon_series = ot.Load_Variable(expt_folder,'Spon_Before.pkl').reset_index(drop = True)
orien_series = ac.Z_Frames['1-007']
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14,4),dpi = 180)
cbar_ax = fig.add_axes([.92, .2, .02, .6])
# plot heat maps, and reset frame index.
vmax = 6
vmin = -2
sns.heatmap((orien_series.iloc[1000:1500,:].T),center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax)
sns.heatmap(spon_series.iloc[2000:2500,:].T,center = 0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax)
xticks = np.array([0,100,200,300,400,500])
axes[1].set_xticks(xticks)
axes[1].set_xticklabels([0,100,200,300,400,500])
# Add selected location for compare.
from matplotlib.patches import Rectangle
axes[0].add_patch(Rectangle((175,0), 6, 520, fill=False, edgecolor='yellow', lw=1,alpha = 0.8))
axes[1].add_patch(Rectangle((46,0), 6, 520, fill=False, edgecolor='yellow', lw=1,alpha = 0.8))
# set ticks and ticks label.
axes[0].set_title('Stim-induced Response')
axes[1].set_title('Spontaneous Response')
axes[1].set_xlabel(f'Frames')
axes[1].set_ylabel(f'Cells')
axes[0].set_ylabel(f'Cells')
# axes[1].xaxis.set_visible(True)
# axes[1].set_xticks([0,100,200,300,400,500])
plt.show()
#%%##############################3. ADD SHUFFLES##########################################
# Step1, get shuffled spon series. Use phase shuffle.
spon_shuffle = Spon_Shuffler(spon_series,method='phase')
spon_shuffle_frame = pd.DataFrame(spon_shuffle,columns = spon_series.columns,index = spon_series.index)
# Step2, sort cell by orientation preference.
spon_series = ot.Load_Variable(expt_folder,'Spon_Before.pkl').reset_index(drop = True)
orien_series = ac.Z_Frames['1-007']
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
sorted_stim_response = orien_series.T.reindex(sorted_cell_sequence.index)
sorted_spon_response = spon_series.T.reindex(sorted_cell_sequence.index)
sorted_shuffle_response = spon_shuffle_frame.T.reindex(sorted_cell_sequence.index)
# Step3, plot a 3 row heatmap, with shuffle on it.
plt.clf()
plt.cla()
vmax = 6
vmin = -2
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 6),dpi = 180)
cbar_ax = fig.add_axes([.92, .2, .02, .6])
# plot heat maps, and reset frame index.
sns.heatmap((sorted_stim_response.iloc[:,1000:1500].T.reset_index(drop = True).T),center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax)
sns.heatmap(sorted_spon_response.iloc[:,2000:2500].T.reset_index(drop = True).T,center = 0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax)
sns.heatmap(sorted_shuffle_response.iloc[:,2000:2500].T.reset_index(drop = True).T,center = 0,xticklabels=False,yticklabels=False,ax = axes[2],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax)
# set ticks and ticks label.
xticks = np.array([0,100,200,300,400,500])
axes[2].set_xticks(xticks)
axes[2].set_xticklabels([0,100,200,300,400,500])
# Add selected location for compare.
from matplotlib.patches import Rectangle
axes[0].add_patch(Rectangle((175,0), 6, 520, fill=False, edgecolor='yellow', lw=1,alpha = 0.8))
axes[1].add_patch(Rectangle((46,0), 6, 520, fill=False, edgecolor='yellow', lw=1,alpha = 0.8))
axes[2].add_patch(Rectangle((46,0), 6, 520, fill=False, edgecolor='yellow', lw=1,alpha = 0.8))
# set graph titles
# fig.tight_layout(rect=[0, 0, .9, 1])
axes[0].set_title('Stim-induced Response')
axes[1].set_title('Spontaneous Response')
axes[2].set_title('Shuffled Spontaneous Response')
axes[2].set_xlabel(f'Frames')
axes[2].set_ylabel(f'Cells')
axes[1].set_ylabel(f'Cells')
axes[0].set_ylabel(f'Cells')
# axes[1].xaxis.set_visible(True)
# axes[1].set_xticks([0,100,200,300,400,500])
plt.show()
#%%######################### 4. Plot Example Averages#####################################
stim_start_point = 175
spon_start_point = 46
stim_recover = orien_series.loc[1000+stim_start_point:1000+stim_start_point+6].mean(0)
spon_recover = spon_series.loc[2000+spon_start_point:2000+spon_start_point+6].mean(0)
shuffle_recover = spon_shuffle_frame.loc[2000+spon_start_point:2000+spon_start_point+6].mean(0)
stim_recover_map = ac.Generate_Weighted_Cell(stim_recover)
spon_recover_map = ac.Generate_Weighted_Cell(spon_recover)
shuffle_recover_map = ac.Generate_Weighted_Cell(shuffle_recover)

plt.clf()
plt.cla()
vmax = 4
vmin = -3
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(4,9),dpi = 180)
fig.tight_layout()
cbar_ax = fig.add_axes([.9, .3, .05, .4])
sns.heatmap(stim_recover_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax,square=True)
sns.heatmap(spon_recover_map,center=0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax,square=True)
sns.heatmap(shuffle_recover_map,center=0,xticklabels=False,yticklabels=False,ax = axes[2],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax,square=True)
axes[0].set_title('Stim-induced Response')
axes[1].set_title('Spontaneous Response')
axes[2].set_title('Shuffled Response')
plt.show()
