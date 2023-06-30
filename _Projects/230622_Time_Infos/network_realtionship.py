'''
This script will discribe relationship between 2 networks, finding whether supression is a real effect.
'''

#%%
import OS_Tools_Kit as ot
import numpy as np
import pandas as pd
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
import Graph_Operation_Kit as gt
import cv2
import umap
import umap.plot
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from Kill_Cache import kill_all_cache
from sklearn.model_selection import cross_val_score
from sklearn import svm
from Cell_Tools.Cell_Visualization import Cell_Weight_Visualization
import random
import seaborn as sns
from My_Wheels.Cell_Class.Stim_Calculators import Stim_Cells
from My_Wheels.Cell_Class.Format_Cell import Cell
from Cell_Class.Advanced_Tools import *
from Cell_Class.Plot_Tools import *
from scipy.stats import pearsonr
import scipy.stats as stats

wp = r'D:\ZR\_Data_Temp\_All_Cell_Classes\220420_L91'
all_labels = ot.Load_Variable(wp,'spon_svc_labels_0420.pkl')
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
spon_frame = ac.Z_Frames['1-001']
#%% all LE and RE events
LE_series = (all_labels>0)*(all_labels<9)*(all_labels%2==1)
RE_series = (all_labels>0)*(all_labels<9)*(all_labels%2==0)
# LE_series = all_labels == 11
# RE_series = all_labels == 15
LE_events,LE_len = Label_Event_Cutter(LE_series)
RE_events,RE_len = Label_Event_Cutter(RE_series)
# get start time of each LE and RE events
LE_start_times = np.zeros(len(LE_events))
RE_start_times = np.zeros(len(RE_events))
for i in range(len(LE_events)):
    LE_start_times[i] = LE_events[i][0]
for i in range(len(RE_events)):
    RE_start_times[i] = RE_events[i][0]
#%% calculate distance between LE and RE.
LE_nearest = np.zeros(len(LE_events))
RE_nearest = np.zeros(len(RE_events))
for i,c_LE in enumerate(LE_events):
    c_LE_time = c_LE[0]
    c_time_diff = abs(RE_start_times-c_LE_time).min()
    LE_nearest[i] = c_time_diff
for i,c_RE in enumerate(RE_events):
    c_RE_time = c_RE[0]
    c_time_diff = abs(LE_start_times-c_RE_time).min()
    RE_nearest[i] = c_time_diff

#%% calculate least distance of each events.
N = 10000
all_LE_wait_time = np.zeros(len(LE_events)*N)
all_RE_wait_time = np.zeros(len(RE_events)*N)
for j in tqdm(range(N)):
    LE_shuffle = Random_Series_Generator(11554,LE_len)
    RE_shuffle = Random_Series_Generator(11554,RE_len)
    LE_events_shuffle,_ = Label_Event_Cutter(LE_shuffle)
    RE_events_shuffle,_ = Label_Event_Cutter(RE_shuffle)
    # calculate start times first.
    LE_start_times_shuffle = np.zeros(len(LE_events_shuffle))
    RE_start_times_shuffle = np.zeros(len(RE_events_shuffle))
    for i in range(len(LE_events_shuffle)):
        LE_start_times_shuffle[i] = LE_events_shuffle[i][0]
    for i in range(len(RE_events_shuffle)):
        RE_start_times_shuffle[i] = RE_events_shuffle[i][0]
    # get time distance.
    c_LE_nearest_shuffle = np.zeros(len(LE_events_shuffle))
    c_RE_nearest_shuffle = np.zeros(len(RE_events_shuffle))
    for i,c_LE in enumerate(LE_events_shuffle):
        c_LE_time = c_LE[0]
        c_time_diff = abs(RE_start_times_shuffle-c_LE_time).min()
        c_LE_nearest_shuffle[i] = c_time_diff
    for i,c_RE in enumerate(RE_events_shuffle):
        c_RE_time = c_RE[0]
        c_time_diff = abs(LE_start_times_shuffle-c_RE_time).min()
        c_RE_nearest_shuffle[i] = c_time_diff
    # write all start time into all_wait_time variables.
    all_LE_wait_time[j*len(c_LE_nearest_shuffle):(j+1)*len(c_LE_nearest_shuffle)] = c_LE_nearest_shuffle
    all_RE_wait_time[j*len(c_RE_nearest_shuffle):(j+1)*len(c_RE_nearest_shuffle)] = c_RE_nearest_shuffle
    
#%% test whether there are difference.
from scipy.stats import mannwhitneyu
plt.hist(LE_nearest,bins = 30)
plt.hist(all_LE_wait_time,bins = 30)

mannwhitneyu(LE_nearest, all_LE_wait_time)
mannwhitneyu(RE_nearest, all_RE_wait_time)

#%% get average Z value of another stim cells.  Both ON/Off/n.s. included.
od_tunings = ac.all_cell_tunings.loc['OD',:]# t values of OD index.
RE_cells = list(od_tunings[od_tunings<-2].index)
LE_cells = list(od_tunings[od_tunings>2].index)
LE_frames = spon_frame.iloc[LE_series,:]
RE_frames = spon_frame.iloc[RE_series,:]
non_eye_frames = spon_frame.iloc[(LE_series+RE_series)!=True,:]
stim_on_frame = spon_frame.iloc[(all_labels>8),:]
#%% and orientation cells
orien_tunings = ac.all_cell_tunings.loc['Best_Orien',:]
orien0_cells = list(orien_tunings[orien_tunings=='Orien0'].index)
orien225_cells = list(orien_tunings[orien_tunings=='Orien22.5'].index)
orien45_cells = list(orien_tunings[orien_tunings=='Orien45'].index)
orien675_cells = list(orien_tunings[orien_tunings=='Orien67.5'].index)
orien90_cells = list(orien_tunings[orien_tunings=='Orien90'].index)
orien1125_cells = list(orien_tunings[orien_tunings=='Orien112.5'].index)
orien135_cells = list(orien_tunings[orien_tunings=='Orien135'].index)
orien1575_cells = list(orien_tunings[orien_tunings=='Orien157.5'].index)
## Another version, using t value of subgraph to determine tuning. Use cell with t value>2.
# hv_t = ac.Orien_t_graphs['H-V'].loc['t_value',:]
# ao_t = ac.Orien_t_graphs['A-O'].loc['t_value',:]
# hhv_t = ac.Orien_t_graphs['Orien22.5-112.5'].loc['t_value',:]
# aoo_t = ac.Orien_t_graphs['Orien67.5-157.5'].loc['t_value',:]
# orien0_cells = list(hv_t[hv_t>2].index)
# orien225_cells = list(hhv_t[hhv_t>2].index)
# orien45_cells = list(ao_t[ao_t>2].index)
# orien675_cells = list(aoo_t[aoo_t>2].index)
# orien90_cells = list(hv_t[hv_t<-2].index)
# orien1125_cells = list(hhv_t[hhv_t<-2].index)
# orien135_cells = list(ao_t[ao_t<-2].index)
# orien1575_cells = list(aoo_t[aoo_t<-2].index)
# and orientation frames
Orien0_frames = spon_frame.iloc[(all_labels==9),:]
Orien225_frames = spon_frame.iloc[(all_labels==10),:]
Orien45_frames = spon_frame.iloc[(all_labels==11),:]
Orien675_frames = spon_frame.iloc[(all_labels==12),:]
Orien90_frames = spon_frame.iloc[(all_labels==13),:]
Orien1125_frames = spon_frame.iloc[(all_labels==14),:]
Orien135_frames = spon_frame.iloc[(all_labels==15),:]
Orien1575_frames = spon_frame.iloc[(all_labels==16),:]
eye_frames = spon_frame.iloc[(LE_series+RE_series)==True,:]
#%% get LE distributions in LE frames
# first, non-weighted LE Z value.
LE_values_oppo = LE_frames.loc[:,RE_cells].mean(0)
LE_values_on = RE_frames.loc[:,RE_cells].mean(0)
# LE_values_ns = non_eye_frames.loc[:,LE_cells].mean(0)
LE_values_all_stim = stim_on_frame.loc[:,RE_cells].mean(0)
# histplots.
bin_edges = np.linspace(-0.5,3,25)
plt.hist(LE_values_on,bins = bin_edges,alpha = 0.8)
plt.hist(LE_values_oppo,bins = bin_edges,alpha = 0.8)
plt.hist(LE_values_all_stim,bins = bin_edges,alpha = 0.8)
#%% Get orien distributions 
same_orien = np.array(Orien135_frames.loc[:,orien135_cells]).flatten()
oppop_orien = np.array(Orien45_frames.loc[:,orien135_cells]).flatten()
# ns_orien = eye_frames.loc[:,orien1125_cells].mean(0)
ns_orien = np.array(spon_frame.iloc[(all_labels>0)*(all_labels!=11)*(all_labels!=15),:].loc[:,orien135_cells]).flatten()

bin_edges = np.linspace(-5,5,50)
plt.hist(np.random.choice(same_orien,100),bins = bin_edges,alpha = 0.8)
plt.hist(np.random.choice(oppop_orien,100),bins = bin_edges,alpha = 0.8)
plt.hist(np.random.choice(ns_orien,100),bins = bin_edges,alpha = 0.8)
