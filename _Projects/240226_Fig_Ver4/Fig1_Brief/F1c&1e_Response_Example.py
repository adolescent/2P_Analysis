'''
This script Generate Fig 1c, indicating 


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
from Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *


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
#%% ####################### FIG 1C, EXAMPLE RESPONSE CURVE##################################

all_response_frame = pd.DataFrame(columns=['Cell','Frame','Stim_Type','Response'])
def Pandas_Filler(data_frame,c_response_frame,cell_name,stim_name):
    # This function will fill data into the given pandas frame. Specific for 
    frame_num,condition_num = c_response_frame.shape
    for i in range(frame_num):
        for j in range(condition_num):
            data_frame.loc[len(data_frame)] = [cell_name,i,stim_name,c_response_frame[i,j]]
    return data_frame

for i,cc in enumerate(cell_example_list):
    c_LE_response = ac.Stim_Reponse_Dics['OD'][cc]['L_All']
    c_RE_response = ac.Stim_Reponse_Dics['OD'][cc]['R_All']
    c_H_response = ac.Stim_Reponse_Dics['Oriens'][cc]['Orien0.0']
    c_A_response = ac.Stim_Reponse_Dics['Oriens'][cc]['Orien45.0']
    c_V_response = ac.Stim_Reponse_Dics['Oriens'][cc]['Orien90.0']
    c_O_response = ac.Stim_Reponse_Dics['Oriens'][cc]['Orien135.0']
    c_Red_response = ac.Stim_Reponse_Dics['Colors'][cc]['Red']
    c_Green_response = ac.Stim_Reponse_Dics['Colors'][cc]['Green']
    c_Blue_response = ac.Stim_Reponse_Dics['Colors'][cc]['Blue']
    all_response_frame = Pandas_Filler(all_response_frame,c_LE_response,cc,'LE')
    all_response_frame = Pandas_Filler(all_response_frame,c_RE_response,cc,'RE')
    all_response_frame = Pandas_Filler(all_response_frame,c_H_response,cc,'Orien0')
    all_response_frame = Pandas_Filler(all_response_frame,c_A_response,cc,'Orien45')
    all_response_frame = Pandas_Filler(all_response_frame,c_V_response,cc,'Orien90')
    all_response_frame = Pandas_Filler(all_response_frame,c_O_response,cc,'Orien135')
    all_response_frame = Pandas_Filler(all_response_frame,c_Red_response,cc,'Red')
    all_response_frame = Pandas_Filler(all_response_frame,c_Green_response,cc,'Green')
    all_response_frame = Pandas_Filler(all_response_frame,c_Blue_response,cc,'Blue')
#%% Plot maps using subgraphs.
plt.cla()
plt.clf()
# cell_example_list = [13,274,293] # already defined
frame_example_list = ['LE','RE','Orien0','Orien45','Orien90','Orien135','Red','Green','Blue']
fig,ax = plt.subplots(len(cell_example_list),len(frame_example_list),figsize = (13,4),dpi = 180)
fig.tight_layout(h_pad=0.5)
# for axs in ax.flat:
#     axs.set(ylabel='Z Score')
#     axs.label_outer()
# plotter of all graph.
for i,cc in enumerate(cell_example_list):
    cc_group = all_response_frame.groupby('Cell').get_group(cc)
    for j,c_condition in enumerate(frame_example_list):
        c_graph = cc_group.groupby('Stim_Type').get_group(c_condition)
        c_graph = c_graph[c_graph['Frame']<10]
        ax[i,j].set(ylim = (-1.2,4))
        ax[i,j].set_xticks([2,4,6,8])
        sns.lineplot(data = c_graph,x = 'Frame',y = 'Response',ax = ax[i,j])
        ax[i,j].axvspan(xmin = 3,xmax = 6,alpha = 0.2,facecolor='g',edgecolor=None) # fill stim on 
        if i == 2:
            # ax[i,j].hlines(y = -1,xmin = 3,xmax = 6,linewidth=2, color='r')
            ax[i,j].set_xlabel(c_condition)
        if j == 0: # for the first row
            ax[i,j].set_ylabel(f'Z Score')
            ax[i,j].xaxis.set_visible(False) # off x axis
            ax[i,j].spines['top'].set_visible(False)  # off frames
            ax[i,j].spines['right'].set_visible(False)
            ax[i,j].spines['bottom'].set_visible(False)
            ax[i,j].spines['left'].set_visible(True)
            if i == 2:
                ax[i,j].xaxis.set_visible(True)
                ax[i,j].spines['bottom'].set_visible(True)
        elif i ==2:
            ax[i,j].xaxis.set_visible(True)
            ax[i,j].yaxis.set_visible(False)
            ax[i,j].spines['top'].set_visible(False)
            ax[i,j].spines['right'].set_visible(False)
            ax[i,j].spines['left'].set_visible(False)
            ax[i,j].spines['bottom'].set_visible(True)
        else:
            ax[i,j].axis('off')
# add subtitles
for i, title in enumerate(cell_example_list):
    fig.text(0.5, 1-(i*.95)/3,f'Cell {title}', va='center', ha='center', fontsize=12)
        # ax[i,j].set_title(f'Response {c_condition}')
    
#%% ############################# FIG 1e, EXAMPLE SPON STIM #############################
'''
This graph discribe the diff between stim and spon trains. Stim ON is on green bg.
'''
import re
# calculate all stim on locations.
spon_series = c_spon.reset_index(drop = True)
orien_series = ac.Z_Frames['1-007']
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
cols = ['{} Response'.format(col) for col in ['Stimulus Evoked','Spontaneous']]
rows = ['Cell {}'.format(row) for row in cell_example_list]
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 4))
# plt.setp(axes.flat, xlabel='X-label', ylabel='Y-label') # this is x and y label.
pad = 5 # in points
for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
for ax, row in zip(axes[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
fig.tight_layout()
# tight_layout doesn't take these labels into account. We'll need 
# to make some room. These numbers are are manually tweaked. 
# You could automatically calculate them, but it's a pain.
fig.subplots_adjust(left=0.15, top=0.95)
# next, plot graphs on fig above.
# Use time range 1200-1400 frane, and spon series 3000-3200
start_times_ids = np.array(start_times_ids)
start_times_ids = start_times_ids[(start_times_ids>1200)*(start_times_ids<1400)] # get stim on range.
for i,cc in enumerate(cell_example_list): # i cells
    c_spon_series = spon_series[cc][3000:3200]
    c_stim_series = orien_series[cc][1200:1400]
    axes[i,0].set(ylim = (-3,5.5))
    axes[i,1].set(ylim = (-3,5.5))
    axes[i,0].plot(c_stim_series)
    axes[i,1].plot(c_spon_series)
    for j,c_stim in enumerate(start_times_ids):
        axes[i,0].axvspan(xmin = c_stim,xmax = c_stim+3,alpha = 0.2,facecolor='g',edgecolor=None) # fill stim on 
    # beautiful hacking.
    axes[i,1].yaxis.set_visible(False)
    axes[i,0].xaxis.set_visible(False)
    axes[i,1].xaxis.set_visible(False)
    axes[i,0].set_ylabel('Z Score')
    if i ==2: # lase row
        axes[i,0].xaxis.set_visible(True)
        axes[i,1].xaxis.set_visible(True)
        axes[i,0].set_xlabel('Frames')
        axes[i,1].set_xlabel('Frames')
plt.show()

