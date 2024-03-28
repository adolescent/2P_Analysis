'''
This script only show the response dF/F difference in spontaneous and in G16.
We find stin on location to compare with spon.
Use Mean dF/F
'''


#%% Import and initialization
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
from Filters import Signal_Filter_v2


all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
#%%######################### BASIC FUNCTIONS ###################################
def dFF(F_series,method = 'least',prop=0.1): # dFF method can be changed here.
    if method == 'least':
        base_num = int(len(F_series)*prop)
        base_id = np.argpartition(F_series, base_num)[:base_num]
        base = F_series[base_id].mean()
    dff_series = (F_series-base)/base
    return dff_series,base


def Generate_F_Series(ac,runname = '1-001',start_time = 0,stop_time = 999999,HP = 0.005,LP = 0.3):
    acd = ac.all_cell_dic
    acn = ac.acn
    
    stop_time = min(len(acd[1][runname]),stop_time)
    # get all F frames first
    F_frames_all = np.zeros(shape = (stop_time-start_time,len(acn)),dtype='f8')
    for j,cc in enumerate(acn):
        c_series_raw = acd[cc][runname][start_time:stop_time]
        # c_series_all = Signal_Filter(c_series_raw,order=5,filter_para=filter_para)
        c_series_all = Signal_Filter_v2(c_series_raw,order=5,HP_freq=HP,LP_freq=LP,fps=1.301)
        F_frames_all[:,j] = c_series_all
    # then cut ON parts if needed.
    output_series = F_frames_all
    return output_series


def dFF_Matrix(F_matrix,method = 'least',prop=0.1):
    dFF_Matrix = np.zeros(shape = F_matrix.shape,dtype = 'f8')
    for i in range(F_matrix.shape[1]):
        c_F_series = F_matrix[:,i]
        c_dff_series,_ = dFF(c_F_series,method,prop)
        dFF_Matrix[:,i] = c_dff_series
    return dFF_Matrix

def dff_Matrix_Select(dff_Matrix,ac,runname='1-006',part = 'ON'):
    sfa = ac.Stim_Frame_Align
    c_stim = sfa['Run'+runname[2:]]
    c_stim = np.array(c_stim['Original_Stim_Train'])
    if part == 'ON':
        cutted_series = dff_Matrix[np.where(c_stim != -1)[0]]
    elif part == 'OFF':
        cutted_series = dff_Matrix[np.where(c_stim == -1)[0]]
    return cutted_series


#%%######################## 1. STATS OF STIM RATIO#################################
ac_strength = pd.DataFrame(columns = ['Loc','Cell','In_Run','dFF'])
stim_spon_ratio = pd.DataFrame(columns = ['Loc','Cell','Ratio'])
for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    # get spon dff frame
    spon_start = c_spon.index[0]
    spon_end = c_spon.index[-1]
    spon_F_frame = Generate_F_Series(ac,'1-001',spon_start,spon_end+1)
    spon_dff_frame = dFF_Matrix(spon_F_frame)
    spon_dff_avr = spon_dff_frame.mean(0)
    # get stim df frame.
    stim_F_frame = Generate_F_Series(ac,ac.orienrun)
    stim_dff_frame = dFF_Matrix(stim_F_frame)
    stim_dff_frame_on = dff_Matrix_Select(stim_dff_frame,ac,ac.orienrun)
    stim_dff_avr = stim_dff_frame_on.mean(0)
    all_ratios = (spon_dff_avr/stim_dff_avr)
    for j in range(len(ac.acn)):
        ac_strength.loc[len(ac_strength),:] = [cloc_name,j+1,'Spontaneous',spon_dff_avr[j]]
        ac_strength.loc[len(ac_strength),:] = [cloc_name,j+1,'Stimulus_ON',stim_dff_avr[j]]
        stim_spon_ratio.loc[len(stim_spon_ratio)] = [cloc_name,j+1,all_ratios[j]]


#%% ######################### 2. PLOT STIM SPON STRENGTH #######################################
        
# plotable_data = ac_strength[ac_strength['Loc'] == 'L76_18M_220902']
plotable_data = ac_strength
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4,7),dpi = 180,sharex= True)

# 1. Plot dF/F values here
# axes.axvline(x = 1,color = 'gray', linestyle = '--')
hists = sns.histplot(plotable_data,x = 'dFF',ax = axes[0],hue = 'In_Run', stat="density",bins = np.linspace(0,2,50),alpha = 0.7)
axes[0].set_xlim(0,2)
axes[0].legend(['Stimulus ON', 'Spontaneous'],prop = { "size": 10 })
axes[0].title.set_text('Average Response')

# 2. Plot spon_stim_ratio here.
pivoted_df = plotable_data.pivot(index=['Loc', 'Cell'], columns='In_Run', values=['dFF'])
pivoted_df = pivoted_df['dFF']
axes[1].plot([0,2],[0,2],color = 'gray', linestyle = '--')
scatter = sns.scatterplot(data=pivoted_df,x = 'Spontaneous',y = 'Stimulus_ON',s = 3,ax = axes[1])
axes[1].set_xlim(0,2)
axes[1].set_ylim(0,2)
axes[1].title.set_text('Cell dF/F Distribution')
axes[1].set_xlabel('Spontaneous')
axes[1].set_ylabel('Stimulus Onset')
fig.tight_layout()
#%% Plot ver2, use joint plots.
plt.clf()
plt.cla()
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6),dpi = 180,sharex= True)

joint = sns.jointplot(data=pivoted_df,x = 'Spontaneous',y = 'Stimulus_ON',s = 3,xlim = (0,2), ylim = (0,2),marginal_kws=dict(bins=np.linspace(0,2,50)),height = 6,ratio=4)

joint.figure.suptitle('Cell dF/F Distribution',size = 16)
joint.ax_joint.plot([0,2], [0,2],color = 'gray', linestyle = '--')
joint.set_axis_labels('Spontaneous', 'Stimulus Onset', fontsize=14)