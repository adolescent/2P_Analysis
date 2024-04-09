'''
This script aims to answer questions provided in ppt v5

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



all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)

import warnings
warnings.filterwarnings("ignore")

wp = r'D:\_Path_For_Figs\240401_Figs_v6\Fig3_OD_Color_Repeat'
all_repeat_ids = ot.Load_Variable(r'D:\_Path_For_Figs\240401_Figs_v6\All_Spon_Repeats_All.pkl')
all_locs = list(all_repeat_ids.keys())
#%% P1 OD-Orien Ratio of all data points
all_repeat_freq = pd.DataFrame(columns = ['Loc','OD Freq','Orien Freq','Color Freq'])
for i,cloc in enumerate(all_locs):
    c_ids = all_repeat_ids[cloc]
    framenum = len(c_ids)
    od_freq = Event_Counter(c_ids['OD']>0)*1.301/framenum
    orien_freq = Event_Counter(c_ids['Orien']>0)*1.301/framenum
    color_freq = Event_Counter(c_ids['Color']>0)*1.301/framenum
    all_repeat_freq.loc[len(all_repeat_freq)] = [cloc,od_freq,orien_freq,color_freq]

# plot
plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (5,5),dpi = 180)
sns.scatterplot(data = all_repeat_freq,x = 'Orien Freq',y = 'OD Freq',ax = ax,hue = 'Loc',legend = False)
ax.set_title('All Loc Orientation-OD Repeat Frequency')
ax.set_ylabel('OD Frequency (Hz)')
ax.set_xlabel('Orientation Frequency (Hz)')
ax.set_yticks(np.linspace(0,0.12,5))
ax.set_xticks(np.linspace(0,0.2,5))
fig.tight_layout()

#%% P2 Peak analysis

# get all repeat peak info first
all_repeat_peaks = pd.DataFrame(columns = ['Loc','Strength','Width','Network'])

for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    c_spon = np.array(ot.Load_Variable(cloc,'Spon_Before.pkl'))
    c_all_repeat_ids = all_repeat_ids[cloc_name]

    #od frame and strength
    c_repeat = np.array(c_all_repeat_ids['OD']>0)
    c_ids,c_len = Label_Event_Cutter(c_repeat)
    # od_frames = c_spon[od_repeat,:]
    # od_strength = od_frames.mean(1)
    for j,c_len in enumerate(c_len):
        c_repeat = c_spon[c_ids[j],:]
        c_strength = c_repeat.mean(1).max() # use the peak strength
        all_repeat_peaks.loc[len(all_repeat_peaks),:] = [cloc_name,c_strength,c_len,'OD']

    # orientation
    c_repeat = np.array(c_all_repeat_ids['Orien']>0)
    c_ids,c_len = Label_Event_Cutter(c_repeat)
    # od_frames = c_spon[od_repeat,:]
    # od_strength = od_frames.mean(1)
    for j,c_len in enumerate(c_len):
        c_repeat = c_spon[c_ids[j],:]
        c_strength = c_repeat.mean(1).max() # use the peak strength
        all_repeat_peaks.loc[len(all_repeat_peaks),:] = [cloc_name,c_strength,c_len,'Orien']

    # color
    c_repeat = np.array(c_all_repeat_ids['Color']>0)
    c_ids,c_len = Label_Event_Cutter(c_repeat)
    # od_frames = c_spon[od_repeat,:]
    # od_strength = od_frames.mean(1)
    for j,c_len in enumerate(c_len):
        c_repeat = c_spon[c_ids[j],:]
        c_strength = c_repeat.mean(1).max() # use the peak strength
        all_repeat_peaks.loc[len(all_repeat_peaks),:] = [cloc_name,c_strength,c_len,'Color']

all_repeat_peaks['Strength'] = all_repeat_peaks['Strength'].astype('f8')
all_repeat_peaks['Width'] = all_repeat_peaks['Width'].astype('f8')/1.301
#%% P3 Plot All peak infos.
plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (5,5),dpi = 180)
sns.boxenplot(data = all_repeat_peaks,x = 'Network',y = 'Strength',ax = ax,hue = 'Network',legend = False,showfliers=False,width = 0.75)
# ax.set_title('All Loc Orientation-OD Repeat Frequency')
# ax.set_ylabel('OD Frequency (Hz)')
# ax.set_xlabel('Orientation Frequency (Hz)')
# ax.set_yticks(np.linspace(0,0.12,5))
# ax.set_xticks(np.linspace(0,0.2,5))
# ax.set_ylabel('Peak Width (s)')
ax.set_ylabel('Peak Strength (Z Score)')
# ax.set_title('Network Repeat Peak Width',size = 16)
ax.set_title('Network Repeat Peak Strength',size = 16)
fig.tight_layout()

#%% P4 Plot averaged info.
avr_repeat = pd.DataFrame(columns = ['Loc','Network','Width','Strength'])
for i,cloc in enumerate(all_locs):
    cloc_all = all_repeat_peaks.groupby('Loc').get_group(cloc)
    c_od = cloc_all.groupby('Network').get_group('OD')
    c_orien = cloc_all.groupby('Network').get_group('Orien')
    c_color = cloc_all.groupby('Network').get_group('Color')
    
    # save avrs.
    avr_repeat.loc[len(avr_repeat),:] = [cloc,'OD',c_od['Width'].mean(),c_od['Strength'].mean()]
    avr_repeat.loc[len(avr_repeat),:] = [cloc,'Orien',c_orien['Width'].mean(),c_orien['Strength'].mean()]
    avr_repeat.loc[len(avr_repeat),:] = [cloc,'Color',c_color['Width'].mean(),c_color['Strength'].mean()]
# %%Plot avr parts.
plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (5,5),dpi = 180)
sns.scatterplot(data = avr_repeat,s = 50,x = 'Width',y = 'Strength',ax = ax,hue = 'Network',legend=True)
# ax.set_title('All Loc Orientation-OD Repeat Frequency')
# ax.set_ylabel('OD Frequency (Hz)')
# ax.set_xlabel('Orientation Frequency (Hz)')
ax.set_yticks(np.linspace(0.2,1,5))
ax.set_xticks(np.linspace(0.5,3,5))
ax.set_xlabel('Peak Width (s)')
ax.set_ylabel('Peak Strength (Z Score)')
# ax.set_title('Network Repeat Peak Width',size = 16)
ax.set_title('Width-Strength Relation',size = 16)
fig.tight_layout()


