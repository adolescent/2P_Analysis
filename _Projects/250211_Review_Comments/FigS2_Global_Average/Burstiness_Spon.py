'''
Calculate burstiness of each cell in spontaneous response, and get a statements.

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
from sklearn.model_selection import cross_val_score
from sklearn import svm
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
from Cell_Class.Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *
from Review_Fix_Funcs import *
from Filters import Signal_Filter_v2
import warnings

all_path_dic = list(ot.Get_Subfolders(r'D:\#Fig_Data\_All_Spon_Data_V1'))

all_path_dic.pop(4)
all_path_dic.pop(6)
save_path = r'D:\_GoogleDrive_Files\#Figs\#250211_Revision1\FigS2'

#%% calculate burstiness of all cell
from scipy.signal import find_peaks,peak_widths
burstiness = pd.DataFrame(columns = ['Loc','Cell','Burstiness'])
burstiness_rand = pd.DataFrame(columns = ['Loc','Cell','Burstiness'])
waittime_v1 = pd.DataFrame(index = range(10000000),columns = ['Loc','Waittime'])
freqs = []
N_shuffle = 10
counter=0
for i,cloc in enumerate(all_path_dic):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    spon_start = c_spon.index[0]
    spon_end = c_spon.index[-1]
    c_spon = Z_refilter(ac,'1-001',spon_start,spon_end).T
    for j in range(len(ac)):
        c_series = c_spon[:,j]
        peaks,_ = find_peaks(c_series, height=-0.5,distance=5,prominence=1) 
        c_raster = np.zeros(len(c_series))
        c_raster[peaks]=1
        waittimes = np.diff(np.where(c_raster==1)[0])
        for k,c_waittime in enumerate(waittimes):
            waittime_v1.loc[counter] = [cloc_name,c_waittime]
            counter+=1
        cc_bur = Burstiness_Index_JN(c_raster,winnum=160)
        for k in range(N_shuffle):
            c_raster_s = Rand_Series(int(c_raster.sum()),len(c_raster))
            cc_bur_s = Burstiness_Index_JN(c_raster_s,winnum=160)
            burstiness_rand.loc[len(burstiness_rand)] = [cloc_name,j+1,cc_bur_s]
        burstiness.loc[len(burstiness)] = [cloc_name,j+1,cc_bur]
        freqs.append(len(peaks)*1.301/len(c_spon))
freqs = np.array(freqs)
ot.Save_Variable(save_path,'Burstiness_V1',burstiness)
ot.Save_Variable(save_path,'Burstiness_V1_shuffle10',burstiness_rand)
waittime_v1 = waittime_v1.dropna(how='any')
ot.Save_Variable(save_path,'Waittime_V1',waittime_v1)

#%% Do same calculation in V2.
all_path_dic_v2 = list(ot.Get_Subfolders(r'D:\#Fig_Data\_All_Spon_Data_V2'))
burstiness_v2 = pd.DataFrame(columns = ['Loc','Cell','Burstiness'])
burstiness_rand_v2 = pd.DataFrame(columns = ['Loc','Cell','Burstiness'])
waittime_v2 = pd.DataFrame(index = range(10000000),columns = ['Loc','Waittime'])
freqs_v2 = []
N_shuffle = 10
counter=0
for i,cloc in enumerate(all_path_dic_v2):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    spon_start = c_spon.index[0]
    spon_end = c_spon.index[-1]
    c_spon = Z_refilter(ac,'1-001',spon_start,spon_end).T
    for j in range(len(ac)):
        c_series = c_spon[:,j]
        peaks,_ = find_peaks(c_series, height=-0.5,distance=5,prominence=1) 
        c_raster = np.zeros(len(c_series))
        c_raster[peaks]=1
        waittimes = np.diff(np.where(c_raster==1)[0])
        for k,c_waittime in enumerate(waittimes):
            waittime_v2.loc[counter] = [cloc_name,c_waittime]
            counter+=1
        cc_bur = Burstiness_Index_JN(c_raster,winnum=160)
        for k in range(N_shuffle):
            c_raster_s = Rand_Series(int(c_raster.sum()),len(c_raster))
            cc_bur_s = Burstiness_Index_JN(c_raster_s,winnum=160)
            burstiness_rand_v2.loc[len(burstiness_rand_v2)] = [cloc_name,j+1,cc_bur_s]
        burstiness_v2.loc[len(burstiness_v2)] = [cloc_name,j+1,cc_bur]
        freqs_v2.append(len(peaks)*1.301/len(c_spon))
freqs_v2 = np.array(freqs_v2)
ot.Save_Variable(save_path,'Burstiness_V2',burstiness_v2)
ot.Save_Variable(save_path,'Burstiness_V2_shuffle10',burstiness_rand_v2)
waittime_v2 = waittime_v2.dropna(how='any')
ot.Save_Variable(save_path,'Waittime_V2',waittime_v2)

#%% Plot burstiness

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4),dpi = 300,sharex= False)

sns.histplot(data = burstiness,x = 'Burstiness',bins = np.linspace(-0.55,0,40),ax = ax)
ax.axvline(x = burstiness['Burstiness'].mean(),linestyle='--',color = [0.7,0.7,0.7])
ax.set_ylim(0,950)
ax.set_xlim(-0.55,0)
ax.set_yticks([0,200,400,600,800])
ax.set_yticklabels([0,200,400,600,800],fontsize = 12)
ax.set_xticks([-0.5,-0.4,-0.3,-0.2,-0.1,0])
ax.set_xticklabels([-0.5,-0.4,-0.3,-0.2,-0.1,0],fontsize = 12)

ax.set_ylabel('')
ax.set_xlabel('')

fig.savefig(ot.join(save_path,'Fig1Q_Burstiness.png'),bbox_inches='tight')

#%% Plot example cell 
'''
Show example response of a cell on graph.
'''

exploc = all_path_dic[2]
ac = ot.Load_Variable_v2(exploc,'Cell_Class.pkl')
c_spon = ot.Load_Variable(exploc,'Spon_Before.pkl')
spon_start = c_spon.index[0]
spon_end = c_spon.index[-1]
c_spon = Z_refilter(ac,'1-001',spon_start,spon_end).T
exp_cell = c_spon[4700:5350,338]
peaks,_ = find_peaks(exp_cell, height=-0.5,distance=5,prominence=1) 

#%% plot part
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,3.5),dpi = 180,sharex= False)

label_size = 12


ax.plot(exp_cell,c=[0.8,0.6,0.6])
ax.plot(peaks,exp_cell[peaks],'x',c = [0.3,0.8,0.3])
fps = 1.301
ax.set_xlim(0,500*fps)
ax.set_xticks([0*fps,100*fps,200*fps,300*fps,400*fps,500*fps])
ax.set_xticklabels([0,100,200,300,400,500],fontsize = label_size)
ax.set_ylim(-2.1,6)
ax.set_yticks([-2,0,2,4,6])
ax.set_yticklabels([-2,0,2,4,6],fontsize = label_size)

ax.set_ylabel('')
fig.savefig(ot.join(save_path,'Cell338_Example.png'),bbox_inches='tight')