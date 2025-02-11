'''
This script will generate heatmaps of example location, and after that we show Z graphs. Both annotated and un-annotated graphs are shown here.
We also changed cmap into bwr here.

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


wp = r'D:\_All_Spon_Data_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(wp,'Cell_Class.pkl')
spon_series = ot.Load_Variable(wp,'Spon_Before.pkl').reset_index(drop = True)
save_path = r'D:\_GoogleDrive_Files\#Figs\240627_Figs_FF1\Fig1'

import warnings
warnings.filterwarnings("ignore")
#%%
'''
Fig 1D, we generate original Heatmaps, without annotating.
'''
# get spon,stim,shuffle frames.
orien_series = ac.Z_Frames['1-007']
spon_shuffle = Spon_Shuffler(spon_series,method='phase')
spon_shuffle_frame = pd.DataFrame(spon_shuffle,columns = spon_series.columns,index = spon_series.index)
# Sort Orien By Cells actually we sort only by raw data.
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

# cbar_ax = fig.add_axes([1, .35, .01, .3])
# label_size = 14
# title_size = 18
vmax = 4
vmin = -2
#%% Plot color bars seperately.
import matplotlib.colors as mcolors
import matplotlib as mpl

# fig, ax = plt.subplots(figsize=(1, 4))
# # vmin, vmax =  np.nanmin(A), np.nanmax(A)
# cmap = mpl.colormaps['bwr']
# norm = mpl.colors.Normalize(-4,4, clip=[-2,4])  # or vmin, vmax
# # norm = mcolors.CenteredNorm(vcenter=0,halfrange=4.0)
# cbar = fig.colorbar(mpl.cm.ScalarMappable(norm, cmap), ax)
# plt.tight_layout()
# plt.show()
plt.clf()
plt.cla()
data = [[vmin,vmax],[vmin,vmax]]
# Create a heatmap
fig, ax = plt.subplots(figsize = (2,1),dpi = 600)
# fig2, ax2 = plt.subplots()
g = sns.heatmap(data, cmap='bwr', center=0,ax = ax,vmax = vmax,vmin = vmin,cbar_kws={"aspect": 10,"shrink": 1,"orientation": "horizontal"})
# Hide the heatmap itself by setting the visibility of its axes
ax.set_visible(False)
g.collections[0].colorbar.set_ticks([-2,0,4])
g.collections[0].colorbar.set_ticklabels([-2,0,4])
g.collections[0].colorbar.ax.tick_params(labelsize=8)
# g.collections[0].colorbar.aspect(50)
# Create colorbar
# fig.colorbar(ax2.collections[0], ax=ax, orientation='vertical')
fig.tight_layout()
# plt.show()
fig.savefig(ot.join(save_path,'Fig1FGH_Bars.png'))
#%% Plot no bar graphs
plt.clf()
plt.cla()
label_size = 10

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10,8),dpi = 180)
sns.heatmap((sorted_stim_response .iloc[1000:1650,:].T),center = 0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = vmax,vmin = vmin,cbar= False,cmap = 'bwr')
sns.heatmap(sorted_spon_response.iloc[4700:5350,:].T,center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = vmax,vmin = vmin,cbar= False,cmap = 'bwr')
sns.heatmap(sorted_shuffle_response.iloc[4700:5350,:].T,center = 0,xticklabels=False,yticklabels=False,ax = axes[2],vmax = vmax,vmin = vmin,cbar= False,cmap = 'bwr')
# cbar_ax.yaxis.label.set_size(label_size)

# xticks = np.array([0,100,200,300,400,500])
# axes[2].set_xticks(xticks)
# axes[2].set_xticklabels([0,100,200,300,400,500],fontsize = label_size)
from matplotlib.patches import Rectangle
axes[1].add_patch(Rectangle((175,0), 6, 520, fill=False, edgecolor='blue', lw=1,alpha = 0.8))
axes[0].add_patch(Rectangle((461,0), 6, 520, fill=False, edgecolor='blue', lw=1,alpha = 0.8))
# axes[1].add_patch(Rectangle((536,0), 6, 520, fill=False, edgecolor='red', lw=1,alpha = 0.8))
axes[2].add_patch(Rectangle((461,0), 6, 520, fill=False, edgecolor='blue', lw=1,alpha = 0.8))

# axes[1].set_title('Stim-induced Response',size = title_size)
# axes[0].set_title('Spontaneous Response',size = title_size)
# axes[2].set_title('Shuffled Spontaneous Response',size = title_size)

# axes[2].set_xlabel(f'Time (s)',size = label_size)
# axes[2].set_ylabel(f'Cells',size = label_size)
# axes[1].set_ylabel(f'Cells',size = label_size)
# axes[0].set_ylabel(f'Cells',size = label_size)

# annotate cell number on it.
# for i in range(3):
#     # axes[i].set_yticks([0,100,200,300,400,500])
#     axes[i].set_yticks([0,180,360,524])
#     # axes[i].set_yticklabels([0,100,200,300,400,500],rotation = 90,fontsize = 7)
#     axes[i].set_yticklabels([0,180,360,524],rotation = 90,fontsize = 8)

fps = 1.301
axes[2].set_xticks([0*fps,100*fps,200*fps,300*fps,400*fps,500*fps])
axes[2].set_xticklabels([0,100,200,300,400,500],fontsize = label_size)

# fig.tight_layout()
plt.show()
# fig.savefig(ot.join(save_path,'1D_Heatmaps_without_annotate.svg'), bbox_inches='tight')
# fig.savefig(ot.join(save_path,'Fig1FGH_Heatmap.png'))

#%%
'''
Part 2, we generate recovered graph. This will be useful on both version. 
'''

stim_start_point = 175
spon_start_point = 461

stim_recover = orien_series.loc[1000+stim_start_point:1000+stim_start_point+6].mean(0)
spon_recover = spon_series.loc[4700+spon_start_point:4700+spon_start_point+6].mean(0)
shuffle_recover = spon_shuffle_frame.loc[0+spon_start_point:0+spon_start_point+6].mean(0)
stim_recover_map = ac.Generate_Weighted_Cell(stim_recover)
spon_recover_map = ac.Generate_Weighted_Cell(spon_recover)
shuffle_recover_map = ac.Generate_Weighted_Cell(shuffle_recover)

#%% Plot color bar seperetly.
plt.clf()
plt.cla()
vmax = 3
vmin = -2
data = [[vmin, vmax], [vmin, vmax]]
# Create a heatmap
fig, ax = plt.subplots(figsize = (2,1),dpi = 600)
# fig2, ax2 = plt.subplots()
g = sns.heatmap(data, center=0,ax = ax,vmax = vmax,vmin = vmin,cbar_kws={"aspect": 10,"shrink": 1,"orientation": "horizontal"})
# Hide the heatmap itself by setting the visibility of its axes
ax.set_visible(False)
g.collections[0].colorbar.set_ticks([vmin,0,vmax])
g.collections[0].colorbar.set_ticklabels([vmin,0,vmax])
g.collections[0].colorbar.ax.tick_params(labelsize=14)
# g.collections[0].colorbar.aspect(50)
# Create colorbar
# fig.colorbar(ax2.collections[0], ax=ax, orientation='vertical')
plt.show()
# fig.savefig(ot.join(save_path,'Fig1FGH_Example_Cell.svg'))

#%%

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5,8),dpi = 180)
fig.tight_layout()
sns.heatmap(spon_recover_map,center=0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = vmax,vmin = vmin,cbar= False,square=True)
sns.heatmap(stim_recover_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = vmax,vmin = vmin,cbar= False,square=True)
sns.heatmap(shuffle_recover_map,center=0,xticklabels=False,yticklabels=False,ax = axes[2],vmax = vmax,vmin = vmin,cbar= False,square=True)


# axes[0].set_title('Spontaneous Response',size = 11)
# axes[1].set_title('Stim-induced Response',size = 11)
# axes[2].set_title('Shuffled Response',size = 11)
fig.tight_layout()
plt.show()

#%%
'''
Fig 1L FFT Power and Power Average.
'''

def Transfer_Into_Freq(input_matrix,freq_bin = 0.01,fps = 1.301):
    input_matrix = np.array(input_matrix)
    # get raw frame spectrums.
    all_specs = np.zeros(shape = ((input_matrix.shape[0]// 2)-1,input_matrix.shape[1]),dtype = 'f8')
    for i in range(input_matrix.shape[1]):
        c_series = input_matrix[:,i]
        c_fft = np.fft.fft(c_series)
        power_spectrum = np.abs(c_fft)[1:input_matrix.shape[0]// 2] ** 2
        power_spectrum = power_spectrum/power_spectrum.sum()
        all_specs[:,i] = power_spectrum
    
    binnum = int(fps/(2*freq_bin))
    binsize = round(len(all_specs)/binnum)
    binned_freq = np.zeros(shape = (binnum,input_matrix.shape[1]),dtype='f8')
    for i in range(binnum):
        c_bin_freqs = all_specs[i*binsize:(i+1)*binsize,:].sum(0)
        binned_freq[i,:] = c_bin_freqs
    return binned_freq

spon_freqs = Transfer_Into_Freq(spon_series)
orien_freqs = Transfer_Into_Freq(orien_series)

#%% Plot power spectrums.
#%% Color bars first.
plt.clf()
plt.cla()
vmax = 0.2
vmin = 0
data = [[vmin, vmax], [vmin, vmax]]
# Create a heatmap
fig, ax = plt.subplots(figsize = (2,1),dpi = 600)
# fig2, ax2 = plt.subplots()
g = sns.heatmap(data, center=0,cmap = 'bwr',ax = ax,vmax = vmax,vmin = vmin,cbar_kws={"aspect": 5,"shrink": 1,"orientation": "horizontal"})
# Hide the heatmap itself by setting the visibility of its axes
ax.set_visible(False)
g.collections[0].colorbar.set_ticks([vmin,0,vmax])
g.collections[0].colorbar.set_ticklabels([vmin,0,vmax])
g.collections[0].colorbar.ax.tick_params(labelsize=14)
plt.show()


#%%
plt.cla()
plt.clf()
fontsize = 14

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(4,5),dpi = 180,sharex= True)
# cbar_ax = fig.add_axes([0.97, .6, .02, .15])
sns.heatmap(spon_freqs[:30,:].T,center = 0,vmax=0.2,ax = ax[0],cbar=False,xticklabels=False,yticklabels=False,cmap = 'bwr')
# sns.heatmap(spon_freqs[:40,:].T,center = 0,vmax=0.15,ax = ax,cbar_ax= cbar_ax,xticklabels=False,yticklabels=False,cbar_kws={'label': 'Spectral Density'})

#plot global powers.
plotable_power = pd.DataFrame(spon_freqs[:30,:].T).melt(var_name='Freq',value_name='Prop.')

# set ticks.
# ax[0].set_yticks([0,180,360,524])
# axes[i].set_yticklabels([0,100,200,300,400,500],rotation = 90,fontsize = 7)
# ax[0].set_yticklabels([0,180,360,524],rotation = 90,fontsize = fontsize)


# set legend.
# cbar_ax.yaxis.label.set_size(10)
# ax[0].set_ylabel('Cell',size = 12)
# ax[1].set_xlabel('Frequency(Hz)',size = 12)
# ax.set_title('Orientation Stimulus Power Spectrum',size = 14)
# ax[0].set_title('Spontaneous Activity Power Spectrum',size = 12)
sns.lineplot(data = plotable_power,x='Freq',y='Prop.',ax = ax[1])
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_xticks([0,10,20,30])
ax[1].set_xticklabels([0,0.1,0.2,0.3],fontsize = fontsize)
ax[1].set_yticks([0.04,0.08])
ax[1].set_yticklabels([0.04,0.08],fontsize = fontsize)
ax[1].set_ylabel('')
ax[1].set_xlabel('')


fig.tight_layout()
# fig.savefig(ot.join(save_path,'1F_Power_Spectrum_Stim.svg'), bbox_inches='tight')

