'''
This script will calculate overlapping prop. of each network, and compare it with random selection.

'''


#%% Initialization
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
from Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *

wp = r'D:\_Path_For_Figs\240312_Figs_v4_A1\A1_Waittime_Distribution'
all_corrs = ot.Load_Variable(wp,'All_Repeat_Series.pkl')
all_locs = list(all_corrs.keys())

#%% ########################### COUNT OVERLAPPING FREMAES ######################
N_shuffle = 100 
overlapping_frames = pd.DataFrame(index = range(1000000),columns = ['Loc','Network_Pair','Counts','Prop','Data_Type','OD_Prop','Orien_Prop','Color_Prop'])
counter = 0

for i,cloc in enumerate(all_locs):
    
    c_repeats = all_corrs[cloc]
    c_od_series = np.array(c_repeats.loc['OD',:]>0).astype('i4')
    c_orien_series = np.array(c_repeats.loc['Orien',:]>0).astype('i4')
    c_color_series = np.array(c_repeats.loc['Color',:]>0).astype('i4')
    # count number of each repeats 
    od_framenum = c_od_series.sum()
    orien_framenum = c_orien_series.sum()
    color_framenum = c_color_series.sum()

    # get overlapping ratios.
    od_orien_overlap_r = (c_od_series*c_orien_series).sum()
    od_color_overlap_r = (c_od_series*c_color_series).sum()
    orien_color_overlap_r = (c_orien_series*c_color_series).sum()
    frame_num = len(c_od_series)
    # save counts into frame
    all_overlapping = [od_orien_overlap_r,od_color_overlap_r,orien_color_overlap_r]
    for j,c_nets in enumerate(['OD_Orien','OD_Color','Orien_Color']):
        overlapping_frames.loc[counter,:] = [cloc,c_nets,all_overlapping[j],all_overlapping[j]/frame_num,'Real_Data',all_overlapping[j]/od_framenum,all_overlapping[j]/orien_framenum,all_overlapping[j]/color_framenum]
        counter += 1
    # and we count number here.
    for j in tqdm(range(N_shuffle)):
        _,all_od_length = Label_Event_Cutter(c_od_series)
        _,all_orien_length = Label_Event_Cutter(c_orien_series)
        _,all_color_length = Label_Event_Cutter(c_color_series)
        #random generate.
        c_od_trains_shuffle = Random_Series_Generator(frame_num,all_od_length)
        c_orien_trains_shuffle = Random_Series_Generator(frame_num,all_orien_length)
        c_color_trains_shuffle = Random_Series_Generator(frame_num,all_color_length)
        # and calculate random over lappings.
        od_orien_overlap_s = (c_od_trains_shuffle*c_orien_trains_shuffle).sum()
        od_color_overlap_s = (c_od_trains_shuffle*c_color_trains_shuffle).sum()
        orien_color_overlap_s = (c_orien_trains_shuffle*c_color_trains_shuffle).sum()
        all_overlapping_s = [od_orien_overlap_s,od_color_overlap_s,orien_color_overlap_s]
        for k,c_nets in enumerate(['OD_Orien','OD_Color','Orien_Color']):
            overlapping_frames.loc[counter,:] = [cloc,c_nets,all_overlapping_s[k],all_overlapping_s[k]/frame_num,'Shuffled_Data',all_overlapping_s[k]/od_framenum,all_overlapping_s[k]/orien_framenum,all_overlapping_s[k]/color_framenum]
            counter+=1

overlapping_frames = overlapping_frames.dropna(how='any').reset_index(drop=True)
ot.Save_Variable(wp,'Overlapping_Frame_Num',overlapping_frames)
#%% ################ REAL OVERLAPPING vs RANDOM OVERLAPPING #######################

plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6),dpi = 180)

sns.boxplot(data = overlapping_frames,x = 'Network_Pair',y = 'Prop',hue = 'Data_Type',width=0.5,ax = ax)
ax.set_title('Overlapped Frame Propotion',size = 16)

#%% ###################### COHERENCE CALCULATION #####################
from scipy import signal
all_coherence = {}
n_gap = 64
all_coherence_frame = pd.DataFrame(columns = ['Loc','Pair','Freq','Data_Type','Coherence'],index = range(2000000))
counter = 0
for i,cloc in enumerate(all_locs):
    
    c_repeats = all_corrs[cloc]
    c_od_series = np.array(c_repeats.loc['OD',:]>0).astype('i4')
    c_orien_series = np.array(c_repeats.loc['Orien',:]>0).astype('i4')
    c_color_series = np.array(c_repeats.loc['Color',:]>0).astype('i4')
    fps = 1.301
    # calculate 
    f, c_od_orien = signal.coherence(c_od_series, c_orien_series, fps, nperseg=n_gap)
    _, c_od_color = signal.coherence(c_od_series, c_color_series, fps, nperseg=n_gap)
    _, c_orien_color = signal.coherence(c_orien_series,c_color_series,fps, nperseg=n_gap)
    # plt.semilogy(f, Cxy)
    # plt.plot(f,Cxy)
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('Coherence')
    # plt.show()
    for j,c_freq in enumerate(f):
        all_coherence_frame.loc[counter,:] = [cloc,'OD-Orien',c_freq,'Real_Data', c_od_orien[j]]
        counter += 1
        all_coherence_frame.loc[counter,:] = [cloc,'OD-Color',c_freq,'Real_Data',  c_od_color[j]]
        counter += 1
        all_coherence_frame.loc[counter,:] = [cloc,'Orien-Color',c_freq,'Real_Data',  c_orien_color[j]]
        counter += 1
    # we also calculate shuffle here.
    for j in tqdm(range(N_shuffle)):

        _,all_od_length = Label_Event_Cutter(c_od_series)
        _,all_orien_length = Label_Event_Cutter(c_orien_series)
        _,all_color_length = Label_Event_Cutter(c_color_series)
        c_od_trains_shuffle = Random_Series_Generator(frame_num,all_od_length)
        c_orien_trains_shuffle = Random_Series_Generator(frame_num,all_orien_length)
        c_color_trains_shuffle = Random_Series_Generator(frame_num,all_color_length)
        # calculate 
        f, c_od_orien_s = signal.coherence(c_od_trains_shuffle,c_orien_trains_shuffle, fps, nperseg=n_gap)
        _, c_od_color_s = signal.coherence(c_od_trains_shuffle,c_color_trains_shuffle, fps, nperseg=n_gap)
        _, c_orien_color_s = signal.coherence(c_orien_trains_shuffle,c_color_trains_shuffle,fps, nperseg=n_gap)
        for k,c_freq in enumerate(f):
            all_coherence_frame.loc[counter,:] = [cloc,'OD-Orien',c_freq,'Shuffled_Data', c_od_orien_s[k]]
            counter += 1
            all_coherence_frame.loc[counter,:] = [cloc,'OD-Color',c_freq,'Shuffled_Data',  c_od_color_s[k]]
            counter += 1
            all_coherence_frame.loc[counter,:] = [cloc,'Orien-Color',c_freq,'Shuffled_Data',  c_orien_color_s[k]]
            counter += 1

all_coherence_frame = all_coherence_frame.dropna(how='any').reset_index(drop=True)
ot.Save_Variable(wp,'All_Coherence_Frame',all_coherence_frame)
#%% ####################### Plot Coherence Plots ##########################

fig,axes = plt.subplots(nrows=1, ncols=3,figsize = (14,5),dpi = 180,sharex= True,sharey= True)
groups = ['OD-Orien','Orien-Color','OD-Color']
# plotable_data = all_coherence_frame.groupby('Pair').get_group('Orien-Color')
# plotable_data = all_coherence_frame.groupby('Pair').get_group('OD-Color')
for i,c_pair in enumerate(groups):
    plotable_data = all_coherence_frame.groupby('Pair').get_group(c_pair)
    sns.lineplot(data = plotable_data,x = 'Freq',y = 'Coherence',hue = 'Data_Type',ax = axes[i])
    axes[i].set_title(f'Network {c_pair}',size = 16)
    axes[i].set_xlabel('Frequency (Hz)',size = 14)
fig.suptitle('Coherence Between Different Networks',y = 0.98,size = 20)
axes[0].set_ylabel('Coherence',size = 14)
fig.tight_layout()

