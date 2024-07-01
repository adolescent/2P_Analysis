'''
This mainly focus on Graph refinement.

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
from Filters import Signal_Filter_v2
import warnings
warnings.filterwarnings("ignore")

expt_folder = r'D:\_All_Spon_Data_V1\L76_18M_220902'
save_path = r'D:\_GoogleDrive_Files\#Figs\240627_Figs_FF1\Fig1'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
c_spon = ot.Load_Variable(expt_folder,'Spon_Before.pkl')

#%%
'''
Fig 1c, example cell response. Use dF/F of cells.
'''

# Step1, re generate data dF/F series
cell_example_list = [47,322,338]
spon_series = pd.DataFrame(ac.Get_dFF_Frames('1-001',0.1,8500,13852))
orien_series = pd.DataFrame(ac.Get_dFF_Frames(ac.orienrun,0.1))

# Get Stim on ids
import re

# calculate all stim on locations.
# spon_series = c_spon.reset_index(drop = True)
# orien_series = ac.Z_Frames['1-007']
ac.Stim_Frame_Align['Run007']
stim_od_ids = (np.array(ac.Stim_Frame_Align['Run007']['Original_Stim_Train'])>0).astype('i4')
# use regular expression to get continious series.
series_str = ''.join(map(str,stim_od_ids))
matches = re.finditer('1+', series_str) # this will get all start time of data.
start_times_ids = []
for match in matches:
    start_times_ids.append(match.start())

#%% Plot parts
plt.cla()
plt.clf()
fontsize = 20
# annotate cell and title.
# cols = ['{} Response'.format(col) for col in ['Spontaneous','Stimulus Evoked']]
rows = ['Cell {}'.format(row) for row in cell_example_list]
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 6),sharey = 'row',sharex = 'col',dpi = 300)
# plt.setp(axes.flat, xlabel='X-label', ylabel='Y-label') # this is x and y label.

## Suptitle of Each Columns
pad = 5 # in points
# for ax, col in zip(axes[0], cols):
#     ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
#                 xycoords='axes fraction', textcoords='offset points',
#                  ha='center', va='baseline',size = 20,weight = 'normal')
    
## Title of each row.
# for ax, row in zip(axes[:,0], rows):
#     ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
#                 xycoords=ax.yaxis.label, textcoords='offset points',
#                  ha='right', va='center',rotation=90,size = 16,weight = 'normal')
# # fig.subplots_adjust(left=0.15, top=0.95)
# we annotate name of cell on the right side.


# plot an zero line for all graphs.
for i in range(3):
    for j in range(2):
        axes[i,j].axhline(0, color='gray', linestyle='--')

# next, plot graphs on fig above.
# Use time range 1200-1400 frane, and spon series 3000-3200
start_times_ids = np.array(start_times_ids)
start_times_ids = start_times_ids[(start_times_ids>1200)*(start_times_ids<1320)] # get stim on range.
for i,cc in enumerate(cell_example_list): # i cells
    c_spon_series = spon_series.loc[3000:3120,cc-1]
    c_stim_series = orien_series.loc[1200:1320,cc-1]
    # axes[i,0].set(ylim = (-3,5.5))
    axes[i,1].set(ylim = (-0.2,3))
    axes[i,1].set_yticks([0,1,2,3])
    axes[i,1].set_yticklabels([0,1,2,3])
    # axes[i,1].set(ylim = (-3,5.5))
    axes[i,1].plot(c_stim_series)
    axes[i,0].plot(c_spon_series)
    for j,c_stim in enumerate(start_times_ids):
        axes[i,1].axvspan(xmin = c_stim,xmax = c_stim+3,alpha = 0.2,facecolor='g',edgecolor=None) # fill stim on 
    # beautiful hacking.
    axes[i,1].yaxis.set_visible(False)
    axes[i,0].xaxis.set_visible(False)
    axes[i,1].xaxis.set_visible(False)
    # axes[i,0].set_ylabel('dF/F',size = 14)
    axes[i,0].yaxis.set_label_coords(-0.04, 0.5) # align y labels.

    if i ==2: # lase row
        axes[i,0].xaxis.set_visible(True)
        axes[i,1].xaxis.set_visible(True)
        # axes[i,0].set_xlabel('Time (s)',size = 14)
        # axes[i,1].set_xlabel('Time (s)',size = 14)

# Set x label into seconds.
axes[2,0].set_xticks(np.arange(2300,2420,20)*1.301)
axes[2,0].set_xticklabels(np.arange(2300,2420,20),fontsize = fontsize)
axes[2,1].set_xticks(np.arange(920,1040,20)*1.301)
axes[2,1].set_xticklabels(np.arange(920,1040,20),fontsize = fontsize)


# for seperate adjust of y label.
axes[0,0].set(ylim = (-0.2,3))
axes[0,0].set_yticks([0,1,2,3])
axes[0,0].set_yticklabels([0,1,2,3],fontsize = fontsize)
axes[1,0].set(ylim = (-0.2,2))
axes[1,0].set_yticks([0,1,2])
axes[1,0].set_yticklabels([0,1,2],fontsize = fontsize)
axes[2,0].set(ylim = (-0.2,1.5))
axes[2,0].set_yticks([0,1,1.5])
axes[2,0].set_yticklabels([0,1,1.5],fontsize = fontsize)
fig.tight_layout()
# plt.show()
fig.savefig(ot.join(save_path,'Fig1DE_Example_Cell.svg'))