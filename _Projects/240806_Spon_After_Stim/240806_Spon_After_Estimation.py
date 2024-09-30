

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


import warnings
warnings.filterwarnings("ignore")

all_path_dic = list(ot.Get_Subfolders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(3) # Run 03 bug
all_path_dic.pop(3) # OD bug
all_path_dic.pop(5) # OD bug

save_path = r'D:\_Path_For_Figs\240806_Spon_After'

#%% Get all before stims
all_before_stim = pd.DataFrame(columns = ['Loc','Midrun'])

for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    if ac.orienrun == '1-007':
        before_stim = 'RD'
    elif ac.orienrun == '1-002':
        before_stim = 'G16'
    # all_before_stim.append(before_stim)
    all_before_stim.loc[len(all_before_stim)] = [cloc_name,before_stim]



#%% Try to get example of before and after stims.
'''
Show example frame of all location's before and after spon
'''
all_expt_frames = {}

for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    before_spon = ac.Z_Frames['1-001'].iloc[-3000:-2350,:] #last 2000 frame of before
    after_spon = ac.Z_Frames['1-003'].iloc[:650,:]
    all_expt_frames[cloc_name] = (before_spon,after_spon)
    # sort all frame by orientation tuning, to make cluster easy to recognize.
    rank_index = pd.DataFrame(index = ac.acn,columns=['Best_Orien','Sort_Index','Sort_Index2'])
    for i,cc in enumerate(ac.acn):
        rank_index.loc[cc]['Best_Orien'] = ac.all_cell_tunings[cc]['Best_Orien']
        if ac.all_cell_tunings[cc]['Best_Orien'] == 'False':
            rank_index.loc[cc]['Sort_Index']=-1
            rank_index.loc[cc]['Sort_Index2']=0
        else:
            orien_tunings = float(ac.all_cell_tunings[cc]['Best_Orien'][5:])
            # rank_index.loc[cc]['Sort_Index'] = np.sin(np.deg2rad(orien_tunings))
            rank_index.loc[cc]['Sort_Index'] = orien_tunings
            rank_index.loc[cc]['Sort_Index2'] = np.cos(np.deg2rad(orien_tunings))
    # actually we sort only by raw data.
    sorted_cell_sequence = rank_index.sort_values(by=['Sort_Index'],ascending=False)
    # sort spon before and after
    sorted_before = before_spon.T.reindex(sorted_cell_sequence.index)
    sorted_after = after_spon.T.reindex(sorted_cell_sequence.index)

    # Plot example graph of each graph,
    plt.clf()
    vmax = 4
    vmin = -2
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,5),dpi = 240,sharex=True)
    sns.heatmap(sorted_before,center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = vmax,vmin = vmin,cbar= False,cmap = 'bwr')
    sns.heatmap(sorted_after,center = 0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = vmax,vmin = vmin,cbar= False,cmap = 'bwr')
    axes[1].set_xticks(np.array([0,100,200,300,400,500])*1.301)
    axes[1].set_xticklabels([0,100,200,300,400,500],fontsize = 14)
    fig.savefig(ot.join(save_path,f'{cloc_name}.png'))
ot.Save_Variable(save_path,'All_Example',all_expt_frames)
#%% 
'''

Get std of all loc to explain after effect increase the cellular difference.
Use std as parameter can easily find diff before and after stim.

'''
loc_diffs = pd.DataFrame(columns = ['Loc','Time','Diff','Mid_stim'])
for i in range(7):
    cloc_name = all_path_dic[i].split('\\')[-1]
    a = np.array(all_expt_frames[cloc_name][1])
    b = a.std(1)
    for j in range(len(b)):
        loc_diffs.loc[len(loc_diffs),:] = [cloc_name,j,b[j],all_before_stim.iloc[i,1]]
#%% Plot part
# plotable = loc_diffs.groupby('Mid_stim').get_group('RD')
plotable = loc_diffs

plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4),dpi = 300)
label_size = 10
sns.lineplot(data = plotable,x = 'Time',y = 'Diff',ax = ax)
# ax.set_xlim(1,650)


#%%
'''
Describe averaged after-effect frame, try to get some pattern?
'''
all_rec_maps = {}
all_rec_resp = {}
for i in tqdm(range(7)):
    cloc = all_path_dic[i]
    # expt_loc = 'L76_18M_220902'
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_after_frame = all_expt_frames[cloc.split('\\')[-1]][1]

    # average first 30 frames, for an example idle map.
    top_resp = c_after_frame.iloc[1:30,:].mean(0)
    graph = ac.Generate_Weighted_Cell(top_resp)
    all_rec_maps[cloc.split('\\')[-1]] = graph
    all_rec_resp[cloc.split('\\')[-1]] = top_resp
    # sns.heatmap(graph,square=True,vmax=3,vmin=-3,center = 0)
#%% Plot parts
    
# first line as rd inter, second line as g16 inter.
id_lists = ['L76_17A_220630','L76_18M_220902','L85_17B_220727','L91_1A_220420','L76_15A_220812','L85_19B_220713','L91_8A_220504']

plt.clf()

vmax = 4
vmin = -2
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12,7),dpi = 300)
label_size = 14

for i in range(7):

    sns.heatmap(all_rec_maps[id_lists[i]],center = 0,xticklabels=False,yticklabels=False,ax = axes[i//4,i%4],vmax = vmax,vmin = vmin,cbar= False,square=True)
    axes[i//4,i%4].set_title(id_lists[i],fontsize = label_size)

axes[1,3].axis('off') # hide last one
fig.tight_layout()
#%% 
'''
Estimate distribution of each loc's resp
'''
after_frame = pd.DataFrame(columns = ['Loc','Cell','After_Response','OD_tune','Orien_tune','Color_tune','F_value'])
for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    ac = ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    all_F = ac.all_cell_F
    c_resp =  all_rec_resp[cloc_name]
    cloc_tunes = ac.all_cell_tunings
    for j in range(len(c_resp)):
        c_od = abs(cloc_tunes.loc['OD',j+1])
        c_orien = cloc_tunes.loc['Orien_index',j+1]
        c_best_color = cloc_tunes.loc['Best_Color',j+1]
        if c_best_color != 'False':
            c_color = cloc_tunes.loc[c_best_color+'-White',j+1]
        else:
            c_color = 0
        after_frame.loc[len(after_frame),:] = [cloc_name,j+1,c_resp[j+1],c_od,c_orien,c_color,all_F[j]]

#%% Plot part
plotable = after_frame

plt.clf()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4),dpi = 300)
sns.scatterplot(data = plotable,y = 'After_Response',x = 'Color_tune',s = 2,lw=0)

ax.set_ylim(-4,4)
# ax.set_xlim(-0.2,3)

