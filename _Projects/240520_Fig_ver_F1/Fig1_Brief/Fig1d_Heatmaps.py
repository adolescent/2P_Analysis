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
save_path = r'D:\_Path_For_Figs\240520_Figs_ver_F1\Fig1_Brief'

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
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10,8),dpi = 180)
cbar_ax = fig.add_axes([1, .35, .01, .3])
label_size = 14
title_size = 18

vmax = 4
vmin = -2
sns.heatmap((sorted_stim_response .iloc[1000:1650,:].T),center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax,cmap = 'bwr')
sns.heatmap(sorted_spon_response.iloc[4700:5350,:].T,center = 0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax,cmap = 'bwr')
sns.heatmap(sorted_shuffle_response.iloc[4700:5350,:].T,center = 0,xticklabels=False,yticklabels=False,ax = axes[2],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax,cbar_kws={'label': 'Z Scored Activity'},cmap = 'bwr')
cbar_ax.yaxis.label.set_size(label_size)

xticks = np.array([0,100,200,300,400,500])
axes[2].set_xticks(xticks)
axes[2].set_xticklabels([0,100,200,300,400,500])
from matplotlib.patches import Rectangle
axes[0].add_patch(Rectangle((175,0), 6, 520, fill=False, edgecolor='blue', lw=1,alpha = 0.8))
axes[1].add_patch(Rectangle((461,0), 6, 520, fill=False, edgecolor='blue', lw=1,alpha = 0.8))
# axes[1].add_patch(Rectangle((536,0), 6, 520, fill=False, edgecolor='red', lw=1,alpha = 0.8))
axes[2].add_patch(Rectangle((461,0), 6, 520, fill=False, edgecolor='blue', lw=1,alpha = 0.8))

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
# fig.savefig(ot.join(save_path,'1D_Heatmaps_without_annotate.svg'), bbox_inches='tight')


#%%
'''
Fig 1D ver 2, we annotate svm repeats on the graph, this method will use svm cluster after it. 
'''

used_orien_ids = np.array(ac.Stim_Frame_Align['Run007']['Original_Stim_Train'][1000:1650])
pcnum = 10
spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=spon_series,sample='Frame',pcnum=pcnum)
analyzer = Classify_Analyzer(ac = ac,model=spon_models,spon_frame=spon_series,od = 0,orien = 1,color = 0,isi = True)
analyzer.Train_SVM_Classifier(C=1)
spon_label = analyzer.spon_label
used_spon_label = spon_label[4700:5350]

# then,get orientation colorbars.
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
import matplotlib as mpl
import colorsys
color_setb = np.zeros(shape = (8,3))
for i,c_orien in enumerate(np.arange(0,180,22.5)):
    c_hue = c_orien/180
    c_lightness = 0.5
    c_saturation = 1
    color_setb[i,:] = colorsys.hls_to_rgb(c_hue, c_lightness, c_saturation)

# rectify colors, get 0-8 ids. 0 as no stim, 1-8 as 8 oriens.
rect_orien = np.zeros(shape = used_orien_ids.shape,dtype='i4')
rect_spon = np.zeros(shape = used_spon_label.shape,dtype='i4')
rect_spon_colors = np.zeros(shape = (len(rect_spon),3))
for i,c_id in enumerate(used_orien_ids):
    if c_id == -1 or c_id == 0:
        rect_orien[i] = 0
    elif c_id <9:
        rect_orien[i] = c_id
    else:
        rect_orien[i] = c_id-8
for i,c_id in enumerate(used_spon_label):
    if c_id != 0:
        rect_spon[i] = c_id-8
    else:
        rect_spon[i] = 0
# get colors.
on_orien_colors = np.zeros(shape = (np.sum(rect_orien>0),3))
on_orien_locs = []
on_spon_colors = np.zeros(shape = (np.sum(rect_spon>0),3))
on_spon_locs = []
counter = 0
for i,c_id in enumerate(rect_orien):
    if c_id>0:
        on_orien_locs.append(i)
        on_orien_colors[counter,:] = color_setb[int(c_id-1)]
        counter += 1
counter = 0
for i,c_id in enumerate(rect_spon):
    if c_id>0:
        on_spon_locs.append(i)
        on_spon_colors[counter,:] = color_setb[int(c_id-1)]
        counter += 1


#%% Add dots on example graphs.
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(11,8),dpi = 180)
cbar_ax = fig.add_axes([0.99, .35, .01, .3])
label_size = 14
title_size = 18
vmax = 4
vmin = -2
# core graphs
sns.heatmap((sorted_stim_response.iloc[1000:1650,:].T),center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax,cmap = 'bwr')
sns.heatmap(sorted_spon_response.iloc[4700:5350,:].T,center = 0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax,cbar_kws={'label': 'Z Scored Activity'},cmap = 'bwr')
sns.heatmap(sorted_shuffle_response.iloc[4700:5350,:].T,center = 0,xticklabels=False,yticklabels=False,ax = axes[2],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax,cbar_kws={'label': 'Z Scored Activity'},cmap = 'bwr')
cbar_ax.yaxis.label.set_size(label_size)

# annotation locs
from matplotlib.patches import Rectangle
axes[0].add_patch(Rectangle((175,0), 6, 520, fill=False, edgecolor='blue', lw=1,alpha = 0.8))
axes[1].add_patch(Rectangle((461,0), 6, 520, fill=False, edgecolor='blue', lw=1,alpha = 0.8))
# axes[1].add_patch(Rectangle((536,0), 6, 520, fill=False, edgecolor='red', lw=1,alpha = 0.8))
axes[2].add_patch(Rectangle((461,0), 6, 520, fill=False, edgecolor='blue', lw=1,alpha = 0.8))



# titles
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
    axes[i].set_ylim(550,0)
fps = 1.301
axes[2].set_xticks([0*fps,100*fps,200*fps,300*fps,400*fps,500*fps])
axes[2].set_xticklabels([0,100,200,300,400,500],fontsize = 8)

# draw repeat points on stim graphs.
# axes[0].scatter(np.array(on_orien_locs)+1,np.ones(len(on_orien_locs))*540, s=3, color=on_orien_colors ,alpha = 0.7) 
# axes[1].scatter(np.array(on_spon_locs)+1,np.ones(len(on_spon_locs))*540, s=3, color=on_spon_colors ,alpha = 0.7)
axes[0].bar(np.array(on_orien_locs)+1,25, bottom=524, width=1.5,color = on_orien_colors, edgecolor='none')
axes[1].bar(np.array(on_spon_locs)+1,25, bottom=524, width=1.5,color = on_spon_colors, edgecolor='none')

# axes[1].scatter(np.arange(650)+1,np.ones(len(rect_spon))*530, s=5, color=rect_spon_colors,alpha = 0.4) 
# axes[1].scatter(0,530,s = 10)
# fig.tight_layout()

# add a color bar onto overall graph.
cax_b = fig.add_axes([1.07, .35, .01, .3])
custom_cmap = mcolors.ListedColormap(color_setb)
bounds = np.arange(0,202.5,22.5)
norm = mpl.colors.BoundaryNorm(bounds, custom_cmap.N)
c_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=custom_cmap),cax=cax_b, label='Orientation')
c_bar.set_ticks(np.arange(0,180,22.5)+11.25)
c_bar.set_ticklabels(np.arange(0,180,22.5))
c_bar.ax.tick_params(size=0)
cax_b.yaxis.label.set_size(label_size)

fig.tight_layout()
plt.show()

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

plt.clf()
plt.cla()
vmax = 3
vmin = -2
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5,8),dpi = 180)
fig.tight_layout()
cbar_ax = fig.add_axes([.77, .4, .02, .2])
sns.heatmap(stim_recover_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax,square=True)
sns.heatmap(spon_recover_map,center=0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax,square=True)
sns.heatmap(shuffle_recover_map,center=0,xticklabels=False,yticklabels=False,ax = axes[2],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax,square=True,cbar_kws={'label': 'Z Scored Activity'})
cbar_ax.yaxis.label.set_size(14)


# axes[0].set_title('Stim-induced Response',size = 11)
# axes[1].set_title('Spontaneous Response',size = 11)
# axes[2].set_title('Shuffled Response',size = 11)
fig.tight_layout()
plt.show()