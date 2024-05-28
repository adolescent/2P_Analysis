


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
save_path = r'D:\_Path_For_Figs\240520_Figs_ver_F1\Fig1_Brief'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
c_spon = ot.Load_Variable(expt_folder,'Spon_Before.pkl')


#%%
'''
Fig 1b-2, This will generate a graph of all cells, used for visualize of all neurons.

'''
all_locs = np.ones(len(ac))
all_cell_graph = ac.Generate_Weighted_Cell(all_locs)
all_avr = ac.global_avr.astype('f8')*0.8
annotate_graph = copy.deepcopy(all_avr/255)
annotate_graph[:,:,2] += all_cell_graph*0.5 # use cv2 need to work with BGR sequence.

plotable_graph = (annotate_graph*255/annotate_graph.max()).astype('u1')
cv2.imwrite(ot.join(save_path,'1b2.png'),plotable_graph)


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
# plt.show()
fig.savefig(ot.join(save_path,'1C_dFF_Stim_Spon_Compare.svg'))

#%%
'''
Fig S1B & Fig 1F - FFT Powers
This graph will show the fft power of each cell, compare between stimulus on and spontaneous. This will show no significant power spectrum. 

For stim and spon,Only the variable plot is different.
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

plt.cla()
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3,5),dpi = 144)
cbar_ax = fig.add_axes([0.95, .35, .02, .3])
sns.heatmap(spon_freqs[:30,:].T,center = 0,vmax=0.2,ax = ax,cbar_ax= cbar_ax,xticklabels=False,yticklabels=False,cbar_kws={'label': 'Power Prop.'},cmap = 'bwr')
# sns.heatmap(spon_freqs[:40,:].T,center = 0,vmax=0.15,ax = ax,cbar_ax= cbar_ax,xticklabels=False,yticklabels=False,cbar_kws={'label': 'Spectral Density'})

ax.set_yticks([0,180,360,524])
# axes[i].set_yticklabels([0,100,200,300,400,500],rotation = 90,fontsize = 7)
ax.set_yticklabels([0,180,360,524],rotation = 90,fontsize = 10)
ax.set_xticks([0,10,20,30])
ax.set_xticklabels([0,0.1,0.2,0.3])

cbar_ax.yaxis.label.set_size(10)
ax.set_ylabel('Cell',size = 12)
ax.set_xlabel('Frequency(Hz)',size = 12)
# ax.set_title('Orientation Stimulus Power Spectrum',size = 14)
ax.set_title('Spontaneous Activity Power Spectrum',size = 12)

fig.savefig(ot.join(save_path,'1F_Power_Spectrum_Stim.svg'), bbox_inches='tight')
