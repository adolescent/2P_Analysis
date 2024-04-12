'''
Use peak find method try to recognize peaks in dF/F series.
bin might be useful.

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
from Filters import Signal_Filter_v2
from Cell_Class.Advanced_Tools import *

expt_folder = r'D:\_Lee_Data\231219_Lee_Data_31fps'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class_Raw.pkl')# ac class is different from our own data!

import warnings
warnings.filterwarnings("ignore")
all_series = ot.Load_Variable(expt_folder,'4bin_Frames.pkl')


#%% Get average dFF plots.

# set single cell response threshold as dF/F > thres to find peaks.
from scipy.signal import find_peaks,peak_widths

thres_std = 0
min_dist = 2

# used_data = all_series['dff']
used_data = np.reshape(all_series['dff'],(6925,4,332)).mean(1)
# plt.plot(used_data[1000:1500,34])

global_on_series = np.zeros(shape=used_data.shape,dtype='f8')
raster_series = np.zeros(shape=used_data.shape,dtype='f8')
for i in tqdm(range(used_data.shape[1])):
    # c_dff = used_data[1000:3000,i]
    c_dff = used_data[:,i]
    c_thres = c_dff.mean()+c_dff.std()*thres_std
    peaks, _ = find_peaks(c_dff, height=c_thres,distance=min_dist*31/4)
    raster_series[peaks,i] = 1
    # plt.plot(c_dff)
    # plt.plot(peaks, c_dff[peaks], "x")
    # and get half width on series.
    c_peak_widths_info = peak_widths(c_dff,peaks,rel_height=0.5)
    on_series = np.zeros(len(c_dff))
    for j,c_peak in enumerate(peaks):
        on_series[round(c_peak_widths_info[2][j]):round(c_peak_widths_info[3][j])] = 1
    global_on_series[:,i] = on_series
#%% Plot raster maps.
plt.clf()
plt.cla()
label_size = 12
title_size = 16
plotable_data = global_on_series[:2000,:].T
# plotable_data = raster_series[:2000,:].T
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,7),dpi = 180,sharex= True)
sns.heatmap(plotable_data,center = 0,xticklabels=False,yticklabels=False,ax = ax[0],cbar=False)

ax[0].set_title('All Cell Raster Plot',size = title_size)
ax[0].set_ylabel('Cells',size = label_size)
ax[1].set_title('Firing Cell Propotion',size = title_size)
ax[1].set_xticks(np.linspace(0,250,6)*31/4)
ax[1].set_xticklabels(np.linspace(0,250,6))
ax[1].set_xlabel('Time (s)',size = label_size)
ax[1].set_ylabel('Propotion',size = label_size)
# plot avr and peaks
avr_series = plotable_data.mean(0)
sns.lineplot(avr_series,ax = ax[1])
peaks, _ = find_peaks(avr_series, height=avr_series.mean(),distance=min_dist*31/4)
ax[1].plot(peaks,avr_series[peaks], "x")

fig.tight_layout()
#%% Get global peaks height and distributions.
global_avr_series = global_on_series.mean(1)
peaks, _ = find_peaks(global_avr_series, height=global_avr_series.mean(),distance=min_dist*31/4)
peak_heights = global_avr_series[peaks]

plt.clf()
plt.cla()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,5),dpi = 180,sharex= True)
sns.histplot(x = peak_heights, ax = ax)
ax.axvline(x = np.median(peak_heights),color = 'gray',linestyle = '--')
ax.text(0.3,30,f'N Peak : {len(peaks)}')
ax.text(0.3,34,f'Peak Median : {np.median(peak_heights):.2f}')
ax.set_xlabel('Active Cell Propotion')

total_time = len(global_avr_series)*4/31
print(f'Ensemble Freq : {len(peaks)/total_time:.5f}')


#%% ######################### RASTER PCA ##################################
# Do pca on raster on series, try to find patterns.
used_data = global_on_series

pcnum = 120
spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=used_data,sample='Frame',pcnum=pcnum)
model_var_ratio = np.array(spon_models.explained_variance_ratio_)
print(f'{pcnum} PCs explain Spontaneous VAR {model_var_ratio[:pcnum].sum()*100:.1f}%')
#%% Recover PCs.
all_pcs_graph = np.zeros(shape = (pcnum,512,512))
clip_std = 5
for i in tqdm(range(pcnum)):
    c_pc = spon_pcs[i,:]
    c_graph = ac.Generate_Weighted_Cell(c_pc)
    c_graph = np.clip(c_graph,-c_graph.std()*clip_std,c_graph.std()*clip_std)
    all_pcs_graph[i,:,:] = c_graph
#%% Plot PCs
plotable_graphs = all_pcs_graph[:15,:,:]
plt.clf()
plt.cla()
value_max = 0.1
value_min = -0.1
font_size = 16
fig,axes = plt.subplots(nrows=3, ncols = 5,figsize = (14,8),dpi = 180)
cbar_ax = fig.add_axes([1, .4, .01, .2])
for i in range(15):
    sns.heatmap(plotable_graphs[i,:,:],center = 0,xticklabels=False,yticklabels=False,ax = axes[i//5,i%5],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True,cbar_kws={'label': 'PC Weight'})
fig.tight_layout()