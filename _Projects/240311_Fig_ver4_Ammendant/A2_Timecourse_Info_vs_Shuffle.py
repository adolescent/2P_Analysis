'''
This script will show the wait time with random selection.
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
from Cell_Class.UMAP_Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *


wp = r'D:\_Path_For_Figs\240312_Figs_v4_A1\A1_Waittime_Distribution'
all_waittime = ot.Load_Variable(wp,'All_Network_Waittime.pkl')
all_repeat_series = ot.Load_Variable(wp,'All_Repeat_Series.pkl')
all_loc_names = list(all_repeat_series.keys())

#%% ######################## BASIC FUNCTIONS ####################

def Start_Time_Finder(series):# BOOL DATA TYPE WILL CAUSE ERROR!
    # Find the indices where the series switches from 0 to 1
    switch_indices = np.where(np.diff(series) == 1)[0] + 1 
    return switch_indices

def Waittime_Calculator(series_start,series_tar): 
    # get response curve here.
    series_start_bin = (np.array(series_start)>0).astype('i4')
    series_tar_bin = (np.array(series_tar)>0).astype('i4')
    # get all start time and wait times
    start_times = Start_Time_Finder(series_start_bin)
    tar_times = Start_Time_Finder(series_tar_bin)

    waittime = []
    all_start_time = []
    for j,c_start in enumerate(start_times):
        c_dist = tar_times - c_start
        c_waittime = c_dist[c_dist>0]
        if len(c_waittime)>0:
            waittime.append(c_waittime.min())
            all_start_time.append(c_start)


    return waittime,all_start_time



#%% ########################## STEP1 CALCULATE SHUFFLED WAITTIME ###########################
N_shuffle = 100
shuffled_waittime = pd.DataFrame(index = range(5000000),columns = ['Loc','Net_Before','Net_After','Waittime','Start_Time'])
counter = 0

for i,cloc in enumerate(all_loc_names):

    c_response_series = all_repeat_series[cloc]
    c_od_trains = (np.array(c_response_series.loc['OD',:])>0).astype('i4')
    c_orien_trains = (np.array(c_response_series.loc['Orien',:])>0).astype('i4')
    c_color_trains = (np.array(c_response_series.loc['Color',:])>0).astype('i4')
    # shuffle N times for waittime disps.
    for j in tqdm(range(N_shuffle)):
        # generate shuffled 3 series first.
        spon_len = len(c_od_trains)
        _,all_od_length = Label_Event_Cutter(c_od_trains)
        _,all_orien_length = Label_Event_Cutter(c_orien_trains)
        _,all_color_length = Label_Event_Cutter(c_color_trains)
        #random generate.
        c_od_trains_shuffle = Random_Series_Generator(spon_len,all_od_length)
        c_orien_trains_shuffle = Random_Series_Generator(spon_len,all_orien_length)
        c_color_trains_shuffle = Random_Series_Generator(spon_len,all_color_length)

        # and calculate 6 random waittimes.
        for k,before_network in enumerate([c_od_trains_shuffle,c_orien_trains_shuffle,c_color_trains_shuffle]):
            c_before_name = ['OD','Orien','Color'][k]
            for l,after_network in enumerate([c_od_trains,c_orien_trains,c_color_trains]):
                c_after_name = ['OD','Orien','Color'][l]
                c_waittime_s,c_waittime_start_s = Waittime_Calculator(before_network,after_network)
                # save wait times in codes above.
                for m,cc_time in enumerate(c_waittime_s):
                    shuffled_waittime.loc[counter,:] = [cloc,c_before_name,c_after_name,cc_time,c_waittime_start_s[m]]
                    counter +=1
        
shuffled_waittime = shuffled_waittime.dropna(how='any').reset_index(drop=True)
ot.Save_Variable(wp,'Shuffled_Waittime_Disp',shuffled_waittime)

#%%###################### STEP2 PLOT SHUFFLED AND REAL ON GRAPHS ##########################
# As we only compare this with random selection, no fit will be done here.

plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8,7),dpi = 180, sharex='col',sharey='row')
network_seq = ['OD','Orien','Color']
vmax = [100,75,80]

for i,c_start in enumerate(network_seq):
    for j,c_end in enumerate(network_seq):
        c_real_disp = np.array(all_waittime.groupby('Net_Before').get_group(c_start).groupby('Net_After').get_group(c_end)['Waittime'])
        c_shuffle_disp = np.array(shuffled_waittime.groupby('Net_Before').get_group(c_start).groupby('Net_After').get_group(c_end)['Waittime'])
        # plot real data and shuffled data.
        axes[i,j].hist(c_shuffle_disp, bins=20, density=True, alpha=0.7,range=[0, vmax[j]], label='Random Series')
        axes[i,j].hist(c_real_disp, bins=20, density=True, alpha=0.7,range=[0, vmax[j]], label='Real Waittime')
        
        axes[i,j].set_xlim(0,vmax[i])
        # axes[i,j].text(vmax[j]*0.6,0.04,f'n real event = {len(c_real_disp)}')
        # axes[i,j].text(vmax[j]*0.6,0.055,f'n real event = {len(c_shuffle_disp)}')



axes[0,2].legend(title = 'Network')
fig.suptitle('Network Waittime vs Random Generated Series',size = 18,y = 0.97)
axes[2,0].set_xlabel('To OD',size = 12)
axes[2,1].set_xlabel('To Orientation',size = 12)
axes[2,2].set_xlabel('To Color',size = 12)
axes[0,0].set_ylabel('From OD',size = 12,rotation = 90)
axes[1,0].set_ylabel('From Orientation',size = 12)
axes[2,0].set_ylabel('From Color',size = 12)

fig.tight_layout()
