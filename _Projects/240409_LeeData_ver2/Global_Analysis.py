'''
This script will describe global basic information of Lee datas.

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

expt_folder = r'D:\_Lee_Data\231219_Lee_Data_31fps'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class_Raw.pkl')# ac class is different from our own data!

import warnings
warnings.filterwarnings("ignore")
#%% ########################## Basic Functions #################################
def dFF(F_series,method = 'least',prop=0.1): # dFF method can be changed here.
    if method == 'least':
        base_num = int(len(F_series)*prop)
        base_id = np.argpartition(F_series, base_num)[:base_num]
        base = F_series[base_id].mean()
    dff_series = (F_series-base)/base
    return dff_series,base

#%% ######################## 0- PREPROCESSING NO BIN APPLIED #####################
# generate 1 binned data.
bin_size = 1
real_fps = 31/bin_size
framenum = len(ac[1]['1-001'])
HP_para = 0.05
LP_para = 4
ignore_size = 1000


binned_num = (framenum-2*ignore_size)//bin_size
binned_F_frame = np.zeros(shape = (binned_num,len(ac)),dtype='f8')
dff_frame = np.zeros(shape = binned_F_frame.shape,dtype='f8')
z_frame = np.zeros(shape = binned_F_frame.shape,dtype='f8')
ac_base = []
for i,cc in tqdm(enumerate(ac.acn)):
    c_series = ac[cc]['1-001']
    if bin_size != 1:
        binned_series = np.reshape(c_series[:len(binned_F_frame)*bin_size],(bin_size,-1)).mean(0)
    else:
        binned_series = c_series
    filted_series = Signal_Filter_v2(binned_series,HP_para,LP_para,real_fps)
    ### remember, the filter caused some problem in first parts of data. so we have to ignore them firts 1000 frames and last 1000 frames.
    dff_series,c_base = dFF(filted_series[ignore_size:-ignore_size])
    z_series = (dff_series-dff_series.mean())/dff_series.std()
    # save
    dff_frame[:,i] = dff_series
    binned_F_frame[:,i] = binned_series[ignore_size:-ignore_size]
    z_frame[:,i] = z_series
    ac_base.append(c_base)

all_series = {}
all_series['dff'] = dff_frame
all_series['F_value'] = binned_F_frame # this frame is real, so it is a little longer.
all_series['Z_score'] = z_frame
all_series['base_F'] = ac_base
ot.Save_Variable(expt_folder,'4bin_Frames',all_series)
#%% ################Plot Example  cell dF/F trains.######################
plt.clf()
plt.cla()
cell_lists = [192,35]
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12,7),dpi = 180,sharex = True)
axes[0].plot(dff_frame[5000:8000,cell_lists[0]-1])
axes[1].plot(dff_frame[5000:8000,cell_lists[1]-1])
axes[0].set_ylabel(f'Cell {cell_lists[0]}')
axes[1].set_ylabel(f'Cell {cell_lists[1]}')
axes[0].set_title('dF/F of Example Cells',size = 16)
axes[1].set_xticks(np.linspace(0,100,6)*real_fps)
axes[1].set_xticklabels(np.linspace(0,100,6))
axes[1].set_xlabel('Time (s)')


#%% ############################# Plot Global Heat Maps #######################

plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,7),dpi = 180,sharex= True)
cbar_ax = fig.add_axes([1, .35, .01, .3])
label_size = 12
title_size = 16
vmin = -3
vmax = 3
# sns.heatmap(dff_frame[5000:6000,:].T,center = 0,xticklabels=False,yticklabels=False,ax = ax,cbar_ax= cbar_ax)
# sns.heatmap(dff_frame[7000:9000,:].T,center = 0,xticklabels=False,yticklabels=False,ax = ax,cbar_ax= cbar_ax,vmin = vmin,vmax = vmax)
sns.heatmap(z_frame[4000:7700,:].T,center = 0,xticklabels=False,yticklabels=False,ax = ax[0],cbar_ax= cbar_ax,vmin = vmin,vmax = vmax)

ax[0].set_title('All Cell Z Score Ensemble',size = title_size)
ax[0].set_ylabel('Cells',size = label_size)
ax[1].set_title('Averaged Global dF/F',size = title_size)
ax[1].set_xticks(np.linspace(0,120,6)*31)
ax[1].set_xticklabels(np.linspace(0,120,6))
ax[1].set_xlabel('Time (s)',size = label_size)
ax[1].set_ylabel('dF/F value',size = label_size)
ax[1].set_ylim(0.2,0.8)
sns.lineplot(dff_frame[4000:7700].mean(1),ax = ax[1])

fig.tight_layout()

#%% ##################### PLOT CELL SPECTRUM ####################

def Transfer_Into_Freq(input_matrix,freq_bin = 0.02,fps = 31):
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

freq_bin = 1
z_frame = z_frame[:(len(z_frame)//freq_bin)*freq_bin,:]
binned_frame = np.reshape(z_frame,(int(len(z_frame)//freq_bin),int(freq_bin),-1)).mean(1)
freqs = Transfer_Into_Freq(binned_frame,fps=31/freq_bin)

#%%
plt.cla()
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,7),dpi = 144)
cbar_ax = fig.add_axes([0.92, .35, .02, .3])

sns.heatmap(freqs[:150,:].T,center = 0,vmax=0.03,ax = ax,cbar_ax= cbar_ax,xticklabels=False,yticklabels=False,cbar_kws={'label': 'Power Prop.'})
ax.set_xticks(np.linspace(0,150,5))
ax.set_xticklabels(np.linspace(0,150,5)*0.02)
ax.set_title('Cell Power Spectrum',size = 16)
ax.set_xlabel('Frequency (Hz)',size = 14)
ax.set_ylabel('Cell',size = 14)
