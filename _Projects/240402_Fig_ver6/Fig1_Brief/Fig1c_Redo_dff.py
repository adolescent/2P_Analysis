'''
The same as ver 5
Only x label changed into seconds.

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

# #%% ########### FIG 1C_New EXAMPLE CELL RESPONSE
# '''
# This graph discribe the diff between stim and spon trains. Stim ON is on green bg.
# '''
# # Step1, calculate dF/F Series (instead of Z scored series.)
# def dFF(F_series,method = 'least',prop=0.1): # dFF method can be changed here.
#     if method == 'least':
#         base_num = int(len(F_series)*prop)
#         base_id = np.argpartition(F_series, base_num)[:base_num]
#       # base = F_series[base_id].mean()
#     dff_series = (F_series-base)/base
#     return dff_series,base


# acn = ac.acn
# spon_series = pd.DataFrame(0.0,columns = acn,index = range(len(c_spon)))
# orien_series = pd.DataFrame(0.0,columns = acn,index = range(len(ac[1]['1-007'])))

# for i,cc in enumerate(acn):
#     c_spon_series = ac[cc]['1-001'][8500:13852]
#     c_orien_series = ac[cc]['1-007'][:]
#     filted_c_spon = Signal_Filter_v2(c_spon_series,0.005,0.3,1.301,True)
#     filted_c_orien = Signal_Filter_v2(c_orien_series,0.005,0.3,1.301,True)
#     c_spon_dff,_ = dFF(filted_c_spon)
#     c_orien_dff,_ = dFF(filted_c_orien)
#     spon_series[cc] =  c_spon_dff
#     orien_series[cc] = c_orien_dff
# # all_F_values = ac.all_cell_dic
    
#%%######################## 1. GET SPON and STIM dFF.
# remember, this id have no names.
spon_series = pd.DataFrame(ac.Get_dFF_Frames('1-001',0.1,8500,13852))
orien_series = pd.DataFrame(ac.Get_dFF_Frames(ac.orienrun,0.1))

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
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 8),sharey = 'row',sharex = 'col')
# plt.setp(axes.flat, xlabel='X-label', ylabel='Y-label') # this is x and y label.
pad = 5 # in points
for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                 ha='center', va='baseline',size = 20,weight = 'normal')
for ax, row in zip(axes[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                 ha='right', va='center',rotation=90,size = 16,weight = 'normal')
# fig.subplots_adjust(left=0.15, top=0.95)
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
    axes[i,0].set(ylim = (-0.2,3))
    axes[i,0].set_yticks([0,1,2,3])
    axes[i,0].set_yticklabels([0,1,2,3])
    # axes[i,1].set(ylim = (-3,5.5))
    axes[i,0].plot(c_stim_series)
    axes[i,1].plot(c_spon_series)
    for j,c_stim in enumerate(start_times_ids):
        axes[i,0].axvspan(xmin = c_stim,xmax = c_stim+3,alpha = 0.2,facecolor='g',edgecolor=None) # fill stim on 
    # beautiful hacking.
    axes[i,1].yaxis.set_visible(False)
    axes[i,0].xaxis.set_visible(False)
    axes[i,1].xaxis.set_visible(False)
    axes[i,0].set_ylabel('dF/F',size = 14)
    if i ==2: # lase row
        axes[i,0].xaxis.set_visible(True)
        axes[i,1].xaxis.set_visible(True)
        axes[i,0].set_xlabel('Time (s)',size = 14)
        axes[i,1].set_xlabel('Time (s)',size = 14)

# Set x label into seconds.
axes[2,1].set_xticks(np.arange(2300,2420,20)*1.301)
axes[2,1].set_xticklabels(np.arange(2300,2420,20))
axes[2,0].set_xticks(np.arange(920,1040,20)*1.301)
axes[2,0].set_xticklabels(np.arange(920,1040,20))


# for seperate adjust of y label.
axes[0,0].set(ylim = (-0.2,3))
axes[0,0].set_yticks([0,1,2,3])
axes[0,0].set_yticklabels([0,1,2,3])
axes[1,0].set(ylim = (-0.2,2))
axes[1,0].set_yticks([0,1,2])
axes[1,0].set_yticklabels([0,1,2])
axes[2,0].set(ylim = (-0.2,1.5))
axes[2,0].set_yticks([0,1,1.5])
axes[2,0].set_yticklabels([0,1,1.5])
fig.tight_layout()
plt.show()

#%%##############################Fig 1F Single Cell Power Spectrum #########
# this part will calculate freq power spectrum.

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

plt.cla()
plt.clf()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,7),dpi = 144)
cbar_ax = fig.add_axes([0.92, .35, .02, .3])

sns.heatmap(orien_freqs[:40,:].T,center = 0,vmax=0.2,ax = ax,cbar_ax= cbar_ax,xticklabels=False,yticklabels=False,cbar_kws={'label': 'Power Prop.'})

# sns.heatmap(spon_freqs[:40,:].T,center = 0,vmax=0.15,ax = ax,cbar_ax= cbar_ax,xticklabels=False,yticklabels=False,cbar_kws={'label': 'Spectral Density'})


ax.set_yticks([0,180,360,524])
# axes[i].set_yticklabels([0,100,200,300,400,500],rotation = 90,fontsize = 7)
ax.set_yticklabels([0,180,360,524],rotation = 90,fontsize = 10)
ax.set_xticks([0,10,20,30,40])
ax.set_xticklabels([0,0.1,0.2,0.3,0.4])

cbar_ax.yaxis.label.set_size(12)
ax.set_ylabel('Cell',size = 14)
ax.set_xlabel('Frequency(Hz)',size = 14)
ax.set_title('Orientation Stimulus Power Spectrum',size = 14)
# ax.set_title('Spontaneous Power Spectrum',size = 14)

