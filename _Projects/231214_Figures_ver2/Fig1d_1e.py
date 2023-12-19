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



#%%########################## FIG 1D, FUNCTION MAPS##################################
od_graph = ac.Generate_Weighted_Cell(ac.OD_t_graphs['OD'].loc['t_value'])
ao_graph = ac.Generate_Weighted_Cell(ac.Orien_t_graphs['A-O'].loc['t_value'])
red_graph = ac.Generate_Weighted_Cell(ac.Color_t_graphs['Red-White'].loc['t_value'])

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 4))
value_max = 50
value_min = -40
cbar_ax = fig.add_axes([.91, .15, .015, .7])

sns.heatmap(od_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[0],square=True,cbar_ax= cbar_ax,vmax = value_max,vmin = value_min)
sns.heatmap(ao_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[1],square=True,cbar_ax= cbar_ax,vmax = value_max,vmin = value_min)
sns.heatmap(red_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[2],square=True,cbar_ax= cbar_ax,vmax = value_max,vmin = value_min)

axes[0].set_title('Left Eye - Right Eye',size = 16)
axes[1].set_title('Orientation 45 - Orientation 135',size = 16)
axes[2].set_title('Red Color - White Color',size = 16)

fig.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=0.88, top=None, hspace=None)
plt.show()

#%%####################### FIG 1E, EXAMPLE SERIES #################################
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
# Here we will get response series of a spon and stim part for all three cells.
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