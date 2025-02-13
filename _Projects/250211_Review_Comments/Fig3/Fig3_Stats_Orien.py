'''
This part redo all svm classifier, and will return stats info for od,orien and color graphs.
Both similarity and repeat frequency included.

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
from Review_Fix_Funcs import *
from Filters import Signal_Filter_v2
import warnings
warnings.filterwarnings("ignore")

all_path_dic = list(ot.Get_Subfolders(r'D:\#Fig_Data\_All_Spon_Data_V1'))

all_path_dic.pop(4)
all_path_dic.pop(6)
save_path = r'D:\_GoogleDrive_Files\#Figs\#250211_Revision1\Fig3'
# if already done, skip step 1 and run this.
# ac_strength = ot.Load_Variable(save_path,'1e_All_Cell_dFF.pkl')

#%%##########################1. Calculate All Loc FrameMaps#################################
explained_var = []
N_shuffle = 10
all_repeat_similarity = pd.DataFrame(columns = ['Loc','Network','Corr','Map_Type','Data_Type'])
all_repeat_freq = pd.DataFrame(columns = ['Loc','Network','Freq','Frame_Prop','Data_Type'])


for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    start = c_spon.index[0]
    end = c_spon.index[-1]
    c_spon = Z_refilter(ac,'1-001',start,end).T
    
    # pcnum = PCNum_Determine(c_spon,sample='Frame',thres = 0.5)
    pcnum = 10
    # all_PCNums.append(pcnum)
    spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=c_spon,sample='Frame',pcnum=pcnum)
    explained_var.append(spon_models.explained_variance_ratio_.sum())
    analyzer = Classify_Analyzer(ac = ac,model=spon_models,spon_frame=c_spon,od = 0, color = 0,orien = 1)
    analyzer.Train_SVM_Classifier()
    # similarity calculator
    analyzer.Similarity_Compare_Average(od = 0,color = 0,orien = 1)
    all_orien_corrs = analyzer.Avr_Similarity
    for j in range(len(all_orien_corrs)):
        c_map_info = all_orien_corrs.iloc[j,:]
        all_repeat_similarity.loc[len(all_repeat_similarity),:] = [cloc_name,c_map_info['Network'],c_map_info['PearsonR'],c_map_info['MapType'],c_map_info['Data']]
    # shuffle calculator
    spon_label = analyzer.spon_label
    g16_frames = (spon_label>0).sum()/len(spon_label)
    g16_events = Event_Counter(spon_label>0)*1.301/len(spon_label)
    all_repeat_freq.loc[len(all_repeat_freq),:] = [cloc_name,'Orien',g16_events,g16_frames,'Real_Data']
    for j in tqdm(range(N_shuffle)):
        spon_s_phase = Spon_Shuffler(c_spon,method='phase')
        spon_s_dim = Spon_Shuffler(c_spon,method='dim')
        # get labels of phase shuffled and dim shuffled frames.
        spon_s_phase_embeddings = spon_models.transform(spon_s_phase)
        spon_s_phase_label = SVC_Fit(analyzer.svm_classifier,spon_s_phase_embeddings,0)
        spon_s_dim_embeddings = spon_models.transform(spon_s_dim)
        spon_s_dim_label = SVC_Fit(analyzer.svm_classifier,spon_s_dim_embeddings,0)
        g16_frames_phase_s = (spon_s_phase_label>0).sum()/len(spon_label)
        g16_events_phase_s = Event_Counter(spon_s_phase_label>0)*1.301/len(spon_label)
        g16_frames_dim_s = (spon_s_dim_label>0).sum()/len(spon_label)
        g16_events_dim_s = Event_Counter(spon_s_dim_label>0)*1.301/len(spon_label)
        # save shuffle 
        all_repeat_freq.loc[len(all_repeat_freq),:] = [cloc_name,'Orien',g16_frames_phase_s,g16_events_phase_s,'Phase_Shuffle']
        all_repeat_freq.loc[len(all_repeat_freq),:] = [cloc_name,'Orien',g16_frames_dim_s,g16_events_dim_s,'Dim_Shuffle']
#%% Plot repeat similarity
plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (2,5),dpi = 180)
ax.axhline(y = 0,color='gray', linestyle='--')
sns.boxplot(data = all_repeat_similarity,x = 'Data_Type',y = 'Corr',hue = 'Network',ax = ax,showfliers = False)
ax.set_title('Stim-like Ensemble Repeat Similarity',size = 10)
ax.set_xlabel('')
ax.set_ylabel('Pearson R')
ax.legend(title = 'Network',fontsize = 5)
ax.set_xticklabels(['Real Data','Random Select'],size = 7)
plt.show()
#%% Plot repeat frequency
plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (2,5),dpi = 180)
ax.axhline(y = 0,color='gray', linestyle='--')

# sns.boxplot(data = all_repeat_freq,x = 'Loc',y = 'Freq',hue = 'Data_Type',ax = ax,showfliers = 0,legend = True)
sns.boxplot(data = all_repeat_freq,x = 'Network',y = 'Freq',hue = 'Data_Type',ax = ax,showfliers = 0,legend = True,hue_order=['Real_Data','Phase_Shuffle'],width=0.5)
ax.set_title('Stim-like Ensemble Repeat Similarity',size = 10)
ax.set_xlabel('')
ax.set_ylabel('Frequency(Hz)')
ax.legend(title = 'Network',fontsize = 5)
# ax.set_xticklabels(['Real Data','Random Select'],size = 7)
# ax.set_xticklabels([''],size = 7)
plt.show()

ot.Save_Variable(save_path,'Orien_Repeat_Similarity',all_repeat_similarity)
ot.Save_Variable(save_path,'Orien_Repeat_Freq',all_repeat_freq)