'''
This script will show dF/F ratio between stimulus and spon.
Data in all location will be used here.

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
from Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *
from Filters import Signal_Filter_v2
import warnings
warnings.filterwarnings("ignore")

all_path_dic = list(ot.Get_Subfolders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
save_path = r'D:\_Path_For_Figs\240520_Figs_ver_F1\Fig1_Brief'
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
    spon_dff_frame = ac.Get_dFF_Frames(runname = '1-001',start = spon_start,stop = spon_end)
    stim_dff_frame = ac.Get_dFF_Frames(runname = ac.orienrun)
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
'''
Fig 1E, we plot Ratios and hisplot od stimulus and spon.
'''
plotable_data = ac_strength
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5,6),dpi = 180,sharex= True)

hists = sns.histplot(plotable_data,x = 'dFF',ax = axes[0],hue = 'In_Run', stat="density",bins = np.linspace(0,2,50),alpha = 0.7)
axes[0].set_xlim(0,2)
axes[0].legend(['Stimulus ON', 'Spontaneous'],prop = { "size": 10 })
axes[0].title.set_text('Average Response')


# 2. Plot spon_stim_ratio here.
pivoted_df = plotable_data.pivot(index=['Loc', 'Cell'], columns='In_Run', values=['dFF'])
pivoted_df = pivoted_df['dFF']
axes[1].plot([0,2],[0,2],color = 'gray', linestyle = '--')
scatter = sns.scatterplot(data=pivoted_df,x = 'Spontaneous',y = 'Stimulus_ON',s = 3,ax = axes[1],linewidth = 0)
axes[1].set_xlim(0,2)
axes[1].set_ylim(0,2)
axes[1].title.set_text('Cell dF/F Distribution')
axes[1].set_xlabel('Spontaneous')
axes[1].set_ylabel('Stimulus ON')
fig.tight_layout()

fig.savefig(ot.join(save_path,'1E_dFF_Stim_vs_Spon.svg'), bbox_inches='tight')
