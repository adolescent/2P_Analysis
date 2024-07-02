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
from sklearn.model_selection import cross_val_score
from sklearn import svm
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import colorsys
import matplotlib as mpl

import warnings
warnings.filterwarnings("ignore")

wp = r'D:\_Path_For_Figs\240520_Figs_ver_F1\Fig4_Cell_In_Spon'
all_pair_corrs = ot.Load_Variable(wp,'All_Pair_Corrs.pkl')


#%%
'''
Fig 4C, we average all locations, and return the good ratio of pairwise correlation.
'''

center = 0.37
vmax = 0.5
vmin = 0.2
used_cmap = 'inferno'
n_bin = 21

all_locname = list(all_pair_corrs.keys())
# example_loc = all_pair_corrs[all_locname[4]]
for i,cloc in enumerate(all_locname):
    if i == 0:
        example_loc = copy.deepcopy(all_pair_corrs[all_locname[i]])
    else:
        example_loc = pd.concat([example_loc,all_pair_corrs[all_locname[i]]])

example_loc['OrienA_bin'] = pd.cut(example_loc['OrienA'], bins=np.linspace(0,180,n_bin), right=False)
example_loc['OrienB_bin'] = pd.cut(example_loc['OrienB'], bins=np.linspace(0,180,n_bin), right=False)

# OD is a little fuzzy, as we need to make +- part have similar bins.
half_bin = int((n_bin-1)/2) # make +- have same bin
od_min = example_loc['OD_B'].min()
# od_min = -2.8
od_max = example_loc['OD_B'].max()-0.2
# od_max = 1.2
od_bins = np.concatenate((np.linspace(od_min,0,half_bin,endpoint=False),(np.linspace(0,od_max,half_bin))))
example_loc['OD_A_bin'] = pd.cut(example_loc['OD_A'], bins=od_bins, right=False)
example_loc['OD_B_bin'] = pd.cut(example_loc['OD_B'], bins=od_bins, right=False)

# Dist neet to use abs 
dist_bins = np.linspace(0,450,n_bin)
example_loc['DistX_bin'] = pd.cut(abs(example_loc['DistX']), bins=dist_bins, right=False)
example_loc['DistY_bin'] = pd.cut(abs(example_loc['DistY']), bins=dist_bins, right=False)

# extend groups to get symetry matrix.
example_loc_sym = copy.deepcopy(example_loc)
temp =  copy.deepcopy(example_loc_sym['OD_B_bin'])
example_loc_sym['OD_B_bin'] = example_loc_sym['OD_A_bin']
example_loc_sym['OD_A_bin'] = temp
temp =  copy.deepcopy(example_loc_sym['OrienB_bin'])
example_loc_sym['OrienB_bin'] = example_loc_sym['OrienA_bin']
example_loc_sym['OrienA_bin'] = temp
example_loc = pd.concat([example_loc,example_loc_sym])

orien_plotable = example_loc.groupby(['OrienA_bin', 'OrienB_bin'], as_index=False)['Corr'].mean()
od_plotable = example_loc.groupby(['OD_A_bin', 'OD_B_bin'], as_index=False)['Corr'].mean()
dist_plotable = example_loc.groupby(['DistX_bin','DistY_bin'], as_index=False)['Corr'].mean()
#%% Plot parts
# bar first

value_max = vmax
value_min = vmin
plt.clf()
plt.cla()
data = [[value_min, value_max], [value_min, value_max]]
# Create a heatmap
fig, ax = plt.subplots(figsize = (2,1),dpi = 600)
# fig2, ax2 = plt.subplots()
g = sns.heatmap(data, center=center,ax = ax,vmax = value_max,vmin = value_min,cbar_kws={"aspect": 5,"shrink": 1,"orientation": 'horizontal'},cmap = used_cmap)
# Hide the heatmap itself by setting the visibility of its axes
ax.set_visible(False)
g.collections[0].colorbar.set_ticks([value_min,value_max])
g.collections[0].colorbar.set_ticklabels([value_min,value_max])
g.collections[0].colorbar.ax.tick_params(labelsize=8)
plt.show()

#%% real graph
plt.clf()
plt.cla()
fontsize = 16

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(17,5),dpi = 300)
# cbar_ax = fig.add_axes([.92, .35, .01, .3])
heatmap_data_orien = orien_plotable.pivot(index='OrienA_bin', columns='OrienB_bin', values='Corr')
heatmap_data_od = od_plotable.pivot(index='OD_A_bin', columns='OD_B_bin', values='Corr')
heatmap_data_dist = dist_plotable.pivot(index='DistX_bin', columns='DistY_bin', values='Corr')
for i,c_map in enumerate([heatmap_data_od,heatmap_data_orien,heatmap_data_dist]):
    c_map.columns = [str(bin.left) for bin in c_map.columns]
    c_map.index = [str(bin.left) for bin in c_map.index]

    
g1 = sns.heatmap(heatmap_data_dist, center =center,vmax = vmax,square= True,ax = axes[0],cbar=False,xticklabels=False,yticklabels=False,cmap=used_cmap,vmin = vmin)
g2 = sns.heatmap(heatmap_data_od, center =center,vmax = vmax,square= True,ax = axes[2],cbar=False,xticklabels=False,yticklabels=False,cmap=used_cmap,vmin = vmin)
g3 = sns.heatmap(heatmap_data_orien, center =center,vmax = vmax,square= True,ax = axes[1],cbar=False,xticklabels=False,yticklabels=False,cmap=used_cmap,vmin = vmin)
g1.set_facecolor('gray')
g2.set_facecolor('gray')
g3.set_facecolor('gray')
# sns.heatmap(heatmap_data, cbar=True,square= True,ax = ax,cbar_ax=cbar_ax)
## legends here.
# orientation
# axes[0].set_xlabel('Distance X')
# axes[0].set_ylabel('Distance Y')
# axes[0].set_title('Correlation vs Distance')
# axes[0].set_xticks([0,6,12,18,24,30])
axes[0].set_xticks([0,4,8,12,16,20])
axes[0].set_xticklabels([0,90,180,270,360,450],fontsize = fontsize)
axes[0].set_yticks([0,4,8,12,16,20])
axes[0].set_yticklabels([0,90,180,270,360,450],fontsize = fontsize)
# od
# axes[1].set_xlabel('OD Tuning A')
# axes[1].set_ylabel('OD Tuning B')
# axes[1].set_title('Correlation vs OD Tuning')

# od_ticks = [0,5,10,15,20,24,29]
od_ticks = [0,4,8,12,16,19]
od_ticks_label = []
for i,c_group in enumerate(od_ticks):
    od_ticks_label.append(np.round(od_bins[c_group],2))
axes[2].set_xticks(od_ticks)
axes[2].set_xticklabels(od_ticks_label,fontsize = fontsize)
axes[2].set_yticks(od_ticks)
axes[2].set_yticklabels(od_ticks_label,fontsize = fontsize)
# add 0 lines on OD.
sns.lineplot(x=[0,0], y=[od_ticks_label[0],od_ticks_label[-1]],color = 'black',ax = axes[1])

orien_ticks = [0,4,8,12,16,20]
orien_ticks_label = []
for i,c_group in enumerate(orien_ticks):
    orien_ticks_label.append(c_group*9)
axes[1].set_xticks(orien_ticks)
axes[1].set_xticklabels(orien_ticks_label,fontsize = fontsize)
axes[1].set_yticks(orien_ticks)
axes[1].set_yticklabels(orien_ticks_label,fontsize = fontsize)
# axes[2].set_xlabel('Orientation A')
# axes[2].set_ylabel('Orientation B')
# axes[2].set_title('Correlation vs Orientation Tuning')
# fig.tight_layout()
for i in range(3):
    # axes[i].xaxis.set_label_position('top') 
    # axes[i].xaxis.tick_top()
    axes[i].invert_yaxis()

fig.tight_layout()
plt.show()

