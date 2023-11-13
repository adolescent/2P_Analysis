'''
This script generate cell response frame on stim and spon data.
'''

#%%

from Cell_Class.Stim_Calculators import Stim_Cells
from Cell_Class.Format_Cell import Cell
import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Plotter.Line_Plotter import EZLine
from tqdm import tqdm
import cv2
import re

work_path = r'D:\_Path_For_Figs\Fig1_Data_Description'
expt_folder = r'D:\_All_Spon_Datas_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
ac.wp = expt_folder
global_avr = cv2.imread(r'D:\_All_Spon_Datas_V1\L76_18M_220902\Global_Average_cai.tif',0) # read as 8 bit gray scale map.
cell_example_list = [47,322,338]

#%% get spon and stim series of cell.
spon_series = ot.Load_Variable(expt_folder,'Spon_Before.pkl').reset_index(drop = True)
orien_series = ac.Z_Frames['1-007']
ac.Stim_Frame_Align['Run007']
stim_od_ids = (np.array(ac.Stim_Frame_Align['Run007']['Original_Stim_Train'])>0).astype('i4')
# use regular expression to get continious series.
series_str = ''.join(map(str,stim_od_ids))
matches = re.finditer('1+', series_str)
start_times_ids = []
for match in matches:
    start_times_ids.append(match.start())
#%% Here we will get response series of a spon and stim part for all three cells.
plt.clf()
plt.cla()
cols = ['{} Response'.format(col) for col in ['Stimulus Evoked','Spontaneous']]
rows = ['Cell {}'.format(row) for row in cell_example_list]
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 4))
# plt.setp(axes.flat, xlabel='X-label', ylabel='Y-label') # this is x and y label.
pad = 5 # in points
for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
for ax, row in zip(axes[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
fig.tight_layout()
# tight_layout doesn't take these labels into account. We'll need 
# to make some room. These numbers are are manually tweaked. 
# You could automatically calculate them, but it's a pain.
fig.subplots_adjust(left=0.15, top=0.95)
# next, plot graphs on fig above.
# Use time range 1200-1400 frane, and spon series 3000-3200
start_times_ids = np.array(start_times_ids)
start_times_ids = start_times_ids[(start_times_ids>1200)*(start_times_ids<1400)] # get stim on range.
for i,cc in enumerate(cell_example_list): # i cells
    c_spon_series = spon_series[cc][3000:3200]
    c_stim_series = orien_series[cc][1200:1400]
    axes[i,0].set(ylim = (-3,5.5))
    axes[i,1].set(ylim = (-3,5.5))
    axes[i,0].plot(c_stim_series)
    axes[i,1].plot(c_spon_series)
    for j,c_stim in enumerate(start_times_ids):
        axes[i,0].axvspan(xmin = c_stim,xmax = c_stim+3,alpha = 0.2,facecolor='g',edgecolor=None) # fill stim on 
    # beautiful hacking.
    axes[i,1].yaxis.set_visible(False)
    axes[i,0].xaxis.set_visible(False)
    axes[i,1].xaxis.set_visible(False)
    axes[i,0].set_ylabel('Z Score')
    if i ==2: # lase row
        axes[i,0].xaxis.set_visible(True)
        axes[i,1].xaxis.set_visible(True)
        axes[i,0].set_xlabel('Frames')
        axes[i,1].set_xlabel('Frames')
plt.show()
#%%###################################################################
# Below is Fig1e, the heat map .
# We use this as Fig 2a.
# Fig 1e, Plot heat map of cells in stim and spon. All cell included.

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
#%% Heatmap plot hackings.
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 4),dpi = 180)
cbar_ax = fig.add_axes([.92, .2, .02, .6])
# plot heat maps, and reset frame index.

sns.heatmap((sorted_stim_response.iloc[:,500:1000].T.reset_index(drop = True).T),center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = 6,vmin = -2,cbar_ax= cbar_ax)
sns.heatmap(sorted_spon_response.iloc[:,2000:2500].T.reset_index(drop = True).T,center = 0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = 6,vmin = -2,cbar_ax= cbar_ax)
# set ticks and ticks label.
xticks = np.array([0,100,200,300,400,500])
axes[1].set_xticks(xticks)
axes[1].set_xticklabels([0,100,200,300,400,500])
# Add selected location for compare.
from matplotlib.patches import Rectangle
axes[0].add_patch(Rectangle((448,0), 6, 520, fill=False, edgecolor='yellow', lw=1,alpha = 0.8))
axes[1].add_patch(Rectangle((407,0), 6, 520, fill=False, edgecolor='yellow', lw=1,alpha = 0.8))
# set graph titles
# fig.tight_layout(rect=[0, 0, .9, 1])
axes[0].set_title('Stim-induced Response')
axes[1].set_title('Spontaneous Response')
axes[1].set_xlabel(f'Frames')
axes[1].set_ylabel(f'Cells')
axes[0].set_ylabel(f'Cells')
# axes[1].xaxis.set_visible(True)
# axes[1].set_xticks([0,100,200,300,400,500])
plt.show()
#%% And we can recover the rectangled frames to generate a stim resposne.
stim_recover = orien_series.loc[500+448:500+448+6].mean(0)
spon_recover = spon_series.loc[2000+407:2000+407+6].mean(0)

stim_recover_map = ac.Generate_Weighted_Cell(stim_recover)
spon_recover_map = ac.Generate_Weighted_Cell(spon_recover)

plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 4),dpi = 180)
fig.tight_layout()
cbar_ax = fig.add_axes([.98, .15, .03, .7])
sns.heatmap(stim_recover_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = 5,vmin = -1,cbar_ax= cbar_ax,square=True)
sns.heatmap(spon_recover_map,center=0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = 5,vmin = -1,cbar_ax= cbar_ax,square=True)
axes[0].set_title('Stim-induced Response')
axes[1].set_title('Spontaneous Response')
plt.show()
#%%######################################################################### 
# Below is fig 1e, functional map. This is quite easy.
OD_tmap = ac.Generate_Weighted_Cell(ac.OD_t_graphs['OD'].loc['t_value'])
AO_tmap = ac.Generate_Weighted_Cell(ac.Orien_t_graphs['A-O'].loc['t_value'])
Red_tmap = ac.Generate_Weighted_Cell(ac.Color_t_graphs['Red-White'].loc['t_value'])

plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 4),dpi = 180)
fig.tight_layout()
cbar_ax = fig.add_axes([.98, .3, .02, .45])
range_max = 45
range_min = -45
sns.heatmap(OD_tmap,center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = range_max,vmin = range_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(AO_tmap,center=0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = range_max,vmin = range_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(Red_tmap,center=0,xticklabels=False,yticklabels=False,ax = axes[2],vmax = range_max,vmin = range_min,cbar_ax= cbar_ax,square=True)
axes[0].set_title('Left Eye - Right Eye')
axes[1].set_title('Orientation 45 - Orientation 135')
axes[2].set_title('Red Color - White Color')
plt.show()
