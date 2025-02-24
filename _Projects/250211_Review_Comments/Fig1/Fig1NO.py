'''
Fig NO calculate the average dRR ratio between stimulus and spontaneous.
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

all_path_dic = list(ot.Get_Subfolders(r'D:\_DataTemp\_Fig_Datas\_All_Spon_Data_V1'))

all_path_dic.pop(4)
all_path_dic.pop(6)
save_path = r'G:\我的云端硬盘\#Figs\#250211_Revision1\Fig1'
# if already done, skip step 1 and run this.
ac_strength = ot.Load_Variable(save_path,'1e_All_Cell_dFF.pkl')
#%%
'''
Step1, we will get all dF/F ratio of all cells, and save them in folders.
'''
ac_strength = pd.DataFrame(columns = ['Loc','Cell','In_Run','dFF'])
stim_spon_ratio = pd.DataFrame(columns = ['Loc','Cell','Ratio'])
for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    spon_start = c_spon.index[0]
    spon_end = c_spon.index[-1]
    spon_dff_frame = dff_refilter(ac,runname = '1-001',start = spon_start,end = spon_end)
    stim_dff_frame = dff_refilter(ac,runname = ac.orienrun)
    # and we select stim ON times only.
    stim_id_train = np.array(ac.Stim_Frame_Align[f'Run{ac.orienrun[2:]}']['Original_Stim_Train'])
    stimon_dff_ids = np.where(stim_id_train>0)[0]
    stimon_dff_frame = stim_dff_frame[stimon_dff_ids,:]
    # and calculate average response.
    spon_dff_avr = spon_dff_frame.mean(0)
    stimon_dff_avr = stimon_dff_frame.mean(0)
    all_ratios = (spon_dff_avr/stimon_dff_avr)
    for j in range(len(ac.acn)):
        ac_strength.loc[len(ac_strength),:] = [cloc_name,j+1,'Spontaneous',spon_dff_avr[j]]
        ac_strength.loc[len(ac_strength),:] = [cloc_name,j+1,'Stimulus_ON',stimon_dff_avr[j]]
        stim_spon_ratio.loc[len(stim_spon_ratio)] = [cloc_name,j+1,all_ratios[j]]

ot.Save_Variable(save_path,'1e_All_Cell_dFF',ac_strength)

#%%
# Done, plot Fig NO


plotable_data = ac_strength
plt.clf()
plt.cla()
fontsize = 14

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4,5),dpi = 300,sharex= False)
fig.subplots_adjust(hspace=0.4)

pivoted_df = plotable_data.pivot(index=['Loc', 'Cell'], columns='In_Run', values=['dFF'])
pivoted_df = pivoted_df['dFF']
axes[0].plot([0,2],[0,2],color = 'gray', linestyle = '--')
scatter = sns.scatterplot(data=pivoted_df,x = 'Spontaneous',y = 'Stimulus_ON',s = 3,ax = axes[0],linewidth = 0,alpha = 0.8,legend=False)
axes[0].set_xlim(0,2)
axes[0].set_ylim(0,2)

# axes[1].title.set_text('Cell dF/F Distribution')
# axes[0].set_xlabel('Spontaneous dF/F')
# axes[0].set_ylabel('Stimulus ON dF/F')
# axes[0].xaxis.tick_top()
# axes[0].xaxis.set_label_position('top') 

hists = sns.histplot(plotable_data,x = 'dFF',ax = axes[1],hue = 'In_Run', stat='percent',bins = np.linspace(0,2,25),alpha = 0.8,common_norm=False,edgecolor='none')
axes[1].set_xlim(0,2)
axes[1].legend(['Stimulus ON', 'Spontaneous'],prop = { "size": 14 })

axes[0].set_yticks([0,0.5,1,1.5,2])
axes[0].set_yticklabels([0,0.5,1,1.5,2],fontsize = fontsize)
axes[1].set_yticks([0,20,40,60])
axes[1].set_yticklabels([0,20,40,60],fontsize = fontsize)
for i in range(2):
    axes[i].set_xticks([0,0.5,1,1.5,2])
    axes[i].set_xticklabels([0,0.5,1,1.5,2],fontsize = fontsize)


axes[0].set_xlabel('')
axes[1].set_xlabel('')
axes[0].set_ylabel('')
axes[1].set_ylabel('')

# fig.savefig(ot.join(save_path,'Fig1NO_dFF_disp.png'),bbox_inches='tight')
#%% Part 2, plot Fig N only, ignore fig O.

plotable_data = ac_strength
plt.clf()
plt.cla()
fontsize = 14

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4,3),dpi = 300,sharex= False)
fig.subplots_adjust(hspace=0.4)

pivoted_df = plotable_data.pivot(index=['Loc', 'Cell'], columns='In_Run', values=['dFF'])
pivoted_df = pivoted_df['dFF']
axes.plot([0,2],[0,2],color = 'gray', linestyle = '--')
scatter = sns.scatterplot(data=pivoted_df,x = 'Spontaneous',y = 'Stimulus_ON',s = 3,ax = axes,linewidth = 0,alpha = 0.8,legend=False)
axes.set_xlim(0,2)
axes.set_ylim(0,2)

# axes[1].title.set_text('Cell dF/F Distribution')
# axes[0].set_xlabel('Spontaneous dF/F')
# axes[0].set_ylabel('Stimulus ON dF/F')
# axes[0].xaxis.tick_top()
# axes[0].xaxis.set_label_position('top') 

axes.set_yticks([0,0.5,1,1.5,2])
axes.set_yticklabels([0,0.5,1,1.5,2],fontsize = fontsize)
axes.set_xticks([0,0.5,1,1.5,2])
axes.set_xticklabels([0,0.5,1,1.5,2],fontsize = fontsize)


axes.set_xlabel('')
axes.set_ylabel('')
fig.savefig(ot.join(save_path,'Fig1N_Only.png'),bbox_inches='tight')


#%%
'''
Fig 1P (Alternative), calculate each cell's stim-spon ratio, and get it's distribution
'''

cell_num = len(ac_strength)//2
spon_stim_ratio = np.zeros(cell_num)
for i in range(cell_num):
    c_spon = ac_strength.iloc[2*i,-1]
    c_stim = ac_strength.iloc[2*i+1,-1]
    spon_stim_ratio[i] = c_spon/c_stim

# plot part
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,3),dpi = 300,sharex= False)

ax.axvline(x = spon_stim_ratio.mean(),linestyle='--',color = [0.7,0.7,0.7])
sns.histplot(spon_stim_ratio,bins = np.linspace(0,2.5,26),ax = ax)

ax.set_xlim(0,2.1)
ax.set_yticks([0,300,600,900])
ax.set_yticklabels([0,300,600,900],fontsize = fontsize)
ax.set_xticks([0,0.5,1,1.5,2])
ax.set_xticklabels([0,0.5,1,1.5,2],fontsize = fontsize)
ax.set_ylabel('')

fig.savefig(ot.join(save_path,'Fig1P_stat_stim_spon.png'),bbox_inches='tight')