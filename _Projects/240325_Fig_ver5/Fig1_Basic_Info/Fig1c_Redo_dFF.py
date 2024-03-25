'''
This script changed the compare series from Z scored data into dF/F data.
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
from Filters import Signal_Filter_v2


work_path = r'D:\_Path_For_Figs\240123_Graph_Revised_v1\Fig1_Revised'
expt_folder = r'D:\_All_Spon_Data_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
ac.wp = expt_folder
global_avr = cv2.imread(f'{expt_folder}\Global_Average_cai.tif',0) # read as 8 bit gray scale map.
c_spon = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
ac.Regenerate_Cell_Graph()

import warnings
warnings.filterwarnings("ignore")
# 3 example cells 
cell_example_list = [47,322,338]


#%% ########### FIG 1C_New EXAMPLE CELL RESPONSE
'''
This graph discribe the diff between stim and spon trains. Stim ON is on green bg.
'''
#%% Step1, calculate dF/F Series (instead of Z scored series.)
acn = ac.acn
spon_series = pd.DataFrame(0.0,columns = acn,index = range(len(c_spon)))
orien_series = pd.DataFrame(0.0,columns = acn,index = range(len(ac[1]['1-007'])))

for i,cc in enumerate(acn):
    c_spon_series = ac[cc]['1-001'][8500:13852]
    c_orien_series = ac[cc]['1-007'][:]
    filted_c_spon = Signal_Filter_v2(c_spon_series,0.005,0.3,1.301,True)
    filted_c_orien = Signal_Filter_v2(c_orien_series,0.005,0.3,1.301,True)
    spon_series[cc] = (filted_c_spon-filted_c_spon.mean())/filted_c_spon.mean()
    orien_series[cc] = (filted_c_orien-filted_c_orien.mean())/filted_c_orien.mean()
# all_F_values = ac.all_cell_dic
# orien_series_F = 
# spon_series_F = 

#%% Step2, plot example response. In dF/F mode.
import re
# calculate all stim on locations.
# spon_series = c_spon.reset_index(drop = True)
# orien_series = ac.Z_Frames['1-007']
ac.Stim_Frame_Align['Run007']
stim_od_ids = (np.array(ac.Stim_Frame_Align['Run007']['Original_Stim_Train'])>0).astype('i4')
# use regular expression to get continious series.
series_str = ''.join(map(str,stim_od_ids))
matches = re.finditer('1+', series_str)
start_times_ids = []
for match in matches:
    start_times_ids.append(match.start())

plt.cla()
plt.clf()


# annotate cell and title.
cols = ['{} Response'.format(col) for col in ['Stimulus Evoked','Spontaneous']]
rows = ['Cell {}'.format(row) for row in cell_example_list]
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 6),sharey = True)
# plt.setp(axes.flat, xlabel='X-label', ylabel='Y-label') # this is x and y label.
pad = 5 # in points
for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
for ax, row in zip(axes[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center',rotation=90,weight='bold')
# fig.subplots_adjust(left=0.15, top=0.95)
    

# next, plot graphs on fig above.
# Use time range 1200-1400 frane, and spon series 3000-3200
start_times_ids = np.array(start_times_ids)
start_times_ids = start_times_ids[(start_times_ids>1200)*(start_times_ids<1400)] # get stim on range.
for i,cc in enumerate(cell_example_list): # i cells
    c_spon_series = spon_series[cc][3000:3200]
    c_stim_series = orien_series[cc][1200:1400]
    # axes[i,0].set(ylim = (-3,5.5))
    axes[i,0].set(ylim = (-0.5,1.5))
    # axes[i,1].set(ylim = (-3,5.5))
    axes[i,0].plot(c_stim_series)
    axes[i,1].plot(c_spon_series)
    for j,c_stim in enumerate(start_times_ids):
        axes[i,0].axvspan(xmin = c_stim,xmax = c_stim+3,alpha = 0.2,facecolor='g',edgecolor=None) # fill stim on 
    # beautiful hacking.
    axes[i,1].yaxis.set_visible(False)
    axes[i,0].xaxis.set_visible(False)
    axes[i,1].xaxis.set_visible(False)
    axes[i,0].set_ylabel('dF/F')
    if i ==2: # lase row
        axes[i,0].xaxis.set_visible(True)
        axes[i,1].xaxis.set_visible(True)
        axes[i,0].set_xlabel('Frames')
        axes[i,1].set_xlabel('Frames')

fig.tight_layout()
plt.show()