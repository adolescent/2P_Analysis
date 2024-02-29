'''
This script generate heat map of all neurons in stim and spon and phase shuffle.
Telling that spon will really repeat functional maps.

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
#%%###################### FIG 1F-A HEATMAPS ##############################
# get spon,stim,shuffle frames.
orien_series = ac.Z_Frames['1-007']
spon_shuffle = Spon_Shuffler(spon_series,method='phase')
spon_shuffle_frame = pd.DataFrame(spon_shuffle,columns = spon_series.columns,index = spon_series.index)
#%% then get best corr locations.
c_model = ot.Load_Variable(wp,'Orien_UMAP_3D_20comp.pkl')
analyzer = UMAP_Analyzer(ac=ac,spon_frame=spon_series,umap_model=c_model,od = 0,orien=1,color=0)
analyzer.Train_SVM_Classifier()
orien1575_spon_locs = np.where(analyzer.spon_label == 16)[0]
corrs = np.zeros(len(orien1575_spon_locs))
c_orienmap = (Select_Frame(frame=spon_series,label=analyzer.spon_label,used_id=[16])[0]).mean(0)
for i,c_loc in enumerate(orien1575_spon_locs):
    c_r,_ = stats.pearsonr(c_orienmap,np.array(spon_series)[c_loc,:])
    corrs[i] = c_r
c_maxid = np.where(corrs == corrs.max())[0]
c_max_sponloc = orien1575_spon_locs[c_maxid]
print(f'Best Location of Orien 157.5 Spon : {c_max_sponloc[0]}')
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
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10,7),dpi = 180)
cbar_ax = fig.add_axes([1, .2, .02, .6])

vmax = 6
vmin = -2
sns.heatmap((sorted_stim_response .iloc[1000:1500,:].T),center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax)
sns.heatmap(sorted_spon_response.iloc[4700:5200,:].T,center = 0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax)
sns.heatmap(sorted_shuffle_response.iloc[4700:5200,:].T,center = 0,xticklabels=False,yticklabels=False,ax = axes[2],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax)


xticks = np.array([0,100,200,300,400,500])
axes[2].set_xticks(xticks)
axes[2].set_xticklabels([0,100,200,300,400,500])
from matplotlib.patches import Rectangle
axes[0].add_patch(Rectangle((175,0), 6, 520, fill=False, edgecolor='yellow', lw=1,alpha = 0.8))
axes[1].add_patch(Rectangle((461,0), 6, 520, fill=False, edgecolor='yellow', lw=1,alpha = 0.8))
axes[2].add_patch(Rectangle((461,0), 6, 520, fill=False, edgecolor='yellow', lw=1,alpha = 0.8))

axes[0].set_title('Stim-induced Response')
axes[1].set_title('Spontaneous Response')
axes[2].set_title('Shuffled Spontaneous Response')

axes[2].set_xlabel(f'Frames')
axes[2].set_ylabel(f'Cells')
axes[1].set_ylabel(f'Cells')
axes[0].set_ylabel(f'Cells')

fig.tight_layout()
plt.show()

#%%########################Fig 1F-2 Recovered Mapd of yellow squares#################
stim_start_point = 175
# spon_start_point = 46
spon_start_point = 461

stim_recover = orien_series.loc[1000+stim_start_point:1000+stim_start_point+6].mean(0)
spon_recover = spon_series.loc[4700+spon_start_point:4700+spon_start_point+6].mean(0)
shuffle_recover = spon_shuffle_frame.loc[0+spon_start_point:0+spon_start_point+6].mean(0)
stim_recover_map = ac.Generate_Weighted_Cell(stim_recover)
spon_recover_map = ac.Generate_Weighted_Cell(spon_recover)
shuffle_recover_map = ac.Generate_Weighted_Cell(shuffle_recover)

plt.clf()
plt.cla()
vmax = 4
vmin = -3
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5,7),dpi = 180)
fig.tight_layout()
cbar_ax = fig.add_axes([.72, .4, .03, .3])
sns.heatmap(stim_recover_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax,square=True)
sns.heatmap(spon_recover_map,center=0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax,square=True)
sns.heatmap(shuffle_recover_map,center=0,xticklabels=False,yticklabels=False,ax = axes[2],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax,square=True)
axes[0].set_title('Stim-induced Response')
axes[1].set_title('Spontaneous Response')
axes[2].set_title('Shuffled Response')
fig.tight_layout()
plt.show()
#%%#####################Fig 1F-2B Recovered Mapd of yellow squares, By Frame#################
raw_orien_run = ot.Load_Variable(f'{wp}\\Orien_Frames_Raw.pkl')
raw_spon_run = ot.Load_Variable(f'{wp}\\Spon_Before_Raw.pkl')
clip_std = 5

def dF_Avrs(raw_frame,start_time,len,clip_std):
    global_avr = raw_frame.mean(0)
    used_frame = raw_frame[start_time:start_time+len,:,:].mean(0)
    dF_avr_frame = used_frame-global_avr
    dF_avr_frame = np.clip(dF_avr_frame,(dF_avr_frame.mean()-dF_avr_frame.std()*clip_std),(dF_avr_frame.mean()+dF_avr_frame.std()*clip_std))

    return dF_avr_frame
#%%
real_df_stim = dF_Avrs(raw_orien_run,1000+stim_start_point,6,clip_std)
real_df_spon = dF_Avrs(raw_spon_run,4700+spon_start_point,6,clip_std)

#%% plot raw dF frames
plt.clf()
plt.cla()
value_max = 600
value_min = -600

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4,7),dpi = 180)
cbar_ax = fig.add_axes([.91, .3, .05, .5])
sns.heatmap(real_df_stim,center = 0,ax = axes[0],xticklabels=False,yticklabels=False,square=True,vmax = value_max,vmin = value_min,cbar_ax= cbar_ax)
sns.heatmap(real_df_spon,center = 0,ax = axes[1],xticklabels=False,yticklabels=False,square=True,vmax = value_max,vmin = value_min,cbar_ax= cbar_ax)

axes[0].set_title('Stim Response',size = 14)
axes[1].set_title('Spontaneous Response',size = 14)
fig.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=0.88, top=None, hspace=None)
plt.show()

#%% Plot dF frames ver2, normalize by each max 10%'s avr.
def get_max_percent_mean(arr,percent = 0.1):
    sorted_arr = np.sort(arr)[::-1]  # Sort the array in descending order
    index = int(len(sorted_arr) * percent)  # Calculate the index for the 10th percentile
    max_10_percent = sorted_arr[:index].mean()  # Extract the elements up to the index
    return max_10_percent
per = 0.01
stim_max = get_max_percent_mean(real_df_stim.flatten(),per)
spon_max = get_max_percent_mean(real_df_spon.flatten(),per)

plt.clf()
plt.cla()
value_max = 1
value_min = -1
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4,7),dpi = 144)
cbar_ax = fig.add_axes([.91, .3, .05, .5])
sns.heatmap(real_df_stim/stim_max,center = 0,ax = axes[0],xticklabels=False,yticklabels=False,square=True,vmax = value_max,vmin = value_min,cbar_ax= cbar_ax)
sns.heatmap(real_df_spon/spon_max,center = 0,ax = axes[1],xticklabels=False,yticklabels=False,square=True,vmax = value_max,vmin = value_min,cbar_ax= cbar_ax)

axes[0].set_title('Stim Response',size = 14)
axes[1].set_title('Spontaneous Response',size = 14)
fig.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=0.88, top=None, hspace=None)
plt.show()
#%% Random select, we select 
num_frames = 6
frame_indices = np.random.choice(raw_spon_run.shape[0], size=num_frames, replace=False)
# Select the frames using the indices
selected_frames = raw_spon_run[frame_indices].mean(0)

plt.clf()
plt.cla()
value_max = 600
value_min = -600
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4,4),dpi = 180)
cbar_ax = fig.add_axes([.91, .3, .05, .5])
sns.heatmap(selected_frames-raw_spon_run.mean(0),center = 0,ax = axes,xticklabels=False,yticklabels=False,square=True,vmax = value_max,vmin = value_min,cbar_ax= cbar_ax)
axes.set_title('Random Selected Spontaneous',size = 14)
fig.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=0.88, top=None, hspace=None)
plt.show()

