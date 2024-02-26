'''
This script will try to compare mixed model and G16 only model in all locations.
We are trying to find the relationship of 2 models.

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
import warnings
warnings.filterwarnings("ignore")

work_path = r'D:\_Path_For_Figs\240123_Graph_Revised_v1\Mix_Orien_Model_Compare'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
#%% ################# 1. GET UMAP CLASSIFY LABELS #########################
all_loc_labels = {}

for i,c_loc in tqdm(enumerate(all_path_dic)):
    c_loc_last = c_loc.split('\\')[-1]
    # calculate real network repeat freq.
    c_spon_frame = ot.Load_Variable(c_loc,'Spon_Before.pkl')
    c_ac = ot.Load_Variable_v2(c_loc,'Cell_Class.pkl')
    c_model_mix = ot.Load_Variable(c_loc,'All_Stim_UMAP_3D_20comp.pkl')
    c_model_g16 = ot.Load_Variable(c_loc,'Orien_UMAP_3D_20comp.pkl')
    analyzer_mix = UMAP_Analyzer(ac = c_ac,umap_model=c_model_mix,spon_frame=c_spon_frame,od = True,orien = True,color = True)
    analyzer_mix.Train_SVM_Classifier()
    analyzer_g16 = UMAP_Analyzer(ac = c_ac,umap_model=c_model_g16,spon_frame=c_spon_frame,od = False,orien = True,color = False)
    analyzer_g16.Train_SVM_Classifier()
    spon_label_mix = analyzer_mix.spon_label
    spon_label_g16 = analyzer_g16.spon_label
    all_loc_labels[c_loc_last] = pd.DataFrame([spon_label_mix,spon_label_g16],index = ['Mix_Model','G16_Model']).T
ot.Save_Variable(work_path,'Mix_G16_Model_Compare',all_loc_labels)
#%%#################### 2.MULTIPLE COMPARES #################################
# Compare repeat number first.
freq_frame = pd.DataFrame(columns = ['Loc','Model_Type','Repeat_Frame','Repeat_Freq'])
all_locnames = list(all_loc_labels.keys())
for i,clocname in enumerate(all_locnames):
    c_labelframes = all_loc_labels[clocname]
    mix_series = np.array(c_labelframes['Mix_Model'])
    g16_series = np.array(c_labelframes['G16_Model'])
    freq_frame.loc[len(freq_frame),:] = [clocname,'Mix_Model',(mix_series>0).mean(),Event_Counter(mix_series)*1.301/len(mix_series)]
    freq_frame.loc[len(freq_frame),:] = [clocname,'G16_Model',(g16_series>0).mean(),Event_Counter(g16_series)*1.301/len(mix_series)]
# plot tools
plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=2,figsize = (5,5),dpi = 180)
# ax.axhline(y = 0,color='gray', linestyle='--')
sns.boxplot(data = freq_frame,y = 'Repeat_Freq',x = 'Model_Type',ax = ax[0],showfliers = False,width = 0.5)
sns.boxplot(data = freq_frame,y = 'Repeat_Frame',x = 'Model_Type',ax = ax[1],showfliers = False,width = 0.5)
# ax.set_title('Stim-like Ensemble Repeat Frequency',size = 10)
for i in range(2):
    ax[i].set_xlabel('')
ax[0].set_ylabel('Repeat Event Frequency(Hz)')
ax[1].set_ylabel('Repeat Frame Proportion')
ax[1].yaxis.set_label_position('right')
ax[1].yaxis.tick_right()
# ax[1].legend(title = 'Network',fontsize = 8)
# ax.set_xticklabels(['Real Data','Random Select'],size = 7)
fig.suptitle('Compare Orientation-Only Model and Mixed Model')
fig.tight_layout()
plt.show()
#%% 2. Get G16 Frames - Mix Frame Alter Matrix.
all_compare_frame = pd.DataFrame(columns = ['Loc','Orien_Model_Subgroup','Mix_Model_Subgroup','Count','Prop'])
for i,clocname in enumerate(all_locnames):
    c_labelframes = all_loc_labels[clocname]
    c_spon_len = len(c_labelframes)
    mix_series = np.array(c_labelframes['Mix_Model'])
    g16_series = np.array(c_labelframes['G16_Model'])
    # get subgroups
    g16_orien = g16_series>0
    g16_isi = g16_series==0
    mix_orien = (mix_series>8)*(mix_series<17)
    mix_od = (mix_series>0)*(mix_series<9)
    mix_color = (mix_series>16)
    mix_isi = (mix_series==0)
    all_compare_frame.loc[len(all_compare_frame),:] = [clocname,'Orientation','Orientation',(g16_orien*mix_orien).sum(),(g16_orien*mix_orien).mean()]
    all_compare_frame.loc[len(all_compare_frame),:] = [clocname,'Orientation','ISI',(g16_orien*mix_isi).sum(),(g16_orien*mix_isi).mean()]
    all_compare_frame.loc[len(all_compare_frame),:] = [clocname,'Orientation','OD',(g16_orien*mix_od).sum(),(g16_orien*mix_od).mean()]
    all_compare_frame.loc[len(all_compare_frame),:] = [clocname,'Orientation','Color',(g16_orien*mix_color).sum(),(g16_orien*mix_color).mean()]

    all_compare_frame.loc[len(all_compare_frame),:] = [clocname,'ISI','ISI',(g16_isi*mix_isi).sum(),(g16_isi*mix_isi).mean()]
    all_compare_frame.loc[len(all_compare_frame),:] = [clocname,'ISI','Orientation',(g16_isi*mix_orien).sum(),(g16_isi*mix_orien).mean()]
    all_compare_frame.loc[len(all_compare_frame),:] = [clocname,'ISI','OD',(g16_isi*mix_od).sum(),(g16_isi*mix_od).mean()]
    all_compare_frame.loc[len(all_compare_frame),:] = [clocname,'ISI','Color',(g16_isi*mix_color).sum(),(g16_isi*mix_color).mean()]

ot.Save_Variable(work_path,'All_Compare_Frame',all_compare_frame)
#%% 3. Get Mix model frames into G16
alter_matrix = all_compare_frame.pivot_table(index = 'Orien_Model_Subgroup',columns='Mix_Model_Subgroup',values=['Prop']).astype('f8')
# alter_matrix = alter_matrix.reindex(index = ['Orientation','ISI'],columns=['Orientation','OD','Color','ISI']).reset_index(drop=True)
# alter_matrix = alter_matrix.reindex(['Orientation','ISI'])
# sns.heatmap(np.array(alter_matrix))
alter_matrix = pd.DataFrame(np.array(alter_matrix),columns = ['Color','Unclassified','Eye','Orientation'],index = ['Unclassified','Orientation'])
alter_matrix = alter_matrix.reindex(index = ['Orientation','Unclassified'],columns=['Orientation','Eye','Color','Unclassified'])
#%% Plot method 1, normalize by row, this show the evolution of G16 model frames.
g16_evo = alter_matrix.div(alter_matrix.sum(axis=1), axis=0).astype('f8')
mix_evo = alter_matrix.div(alter_matrix.sum(axis=0), axis=1).astype('f8')
plt.clf()
plt.cla()
vmax = 1
vmin = 0
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 6),dpi = 180,sharex=True)
cbar_ax = fig.add_axes([.95, .2, .02, .6])

sns.heatmap(g16_evo,annot=True,fmt=".3f",center = 0,ax = axes[0],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax,square=True,linewidth=1,cmap='seismic',linecolor='black')
sns.heatmap(mix_evo,annot=True,fmt=".3f",center = 0,ax = axes[1],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax,square=True,linewidth=1,cmap='seismic',linecolor='black')

# legend changes
axes[0].set_ylabel(f'Class of Orientation Only Model',size = 14)
axes[0].yaxis.set_label_coords(-0.1,0)
axes[1].set_xlabel(f'Class of Mixed Model',size = 14)
# frame
for i in range(2):
    axes[i].axhline(y = 1.99,color = 'black', linewidth = 1) 
    axes[i].axhline(y = 0.01,color = 'black', linewidth = 1) 
    axes[i].axvline(x = 0.01,color = 'black', linewidth = 1)
    axes[i].axvline(x = 3.99,color = 'black', linewidth = 1)

# title
axes[0].set_title(f'Normalize Through Orien-only Model',size = 12)
axes[1].set_title(f'Normalize Through Mixed Model',size = 12)
fig.suptitle('Frame Class in Different Models')
fig.tight_layout()
plt.show()
# %%
