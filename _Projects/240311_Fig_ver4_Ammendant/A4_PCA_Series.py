'''
This will show PCA weight map of example location. We don't expect any difference.

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

import warnings
warnings.filterwarnings("ignore")

wp = r'D:\_All_Spon_Data_V1\L76_18M_220902'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
spon_series = ot.Load_Variable(wp,'Spon_Before.pkl')
# if we need raw frame dF values
# raw_orien_run = ot.Load_Variable(f'{wp}\\Orien_Frames_Raw.pkl')
# raw_spon_run = ot.Load_Variable(f'{wp}\\Spon_Before_Raw.pkl')

#%% ############################# Step0, Calculate PCA Model##################################
# determine num of pcs first.
# pc_num = 10
spon_series = np.array(spon_series)
# pcnum = PCNum_Determine(spon_series,sample='Frame',thres = 0.5)
pcnum = 10
g16_frames,g16_labels = ac.Combine_Frame_Labels(od = 0,color = 0,orien = 1)

spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=spon_series,sample='Frame',pcnum=pcnum)
model_var_ratio = np.array(spon_models.explained_variance_ratio_)
print(f'{pcnum} PCs explain Spontaneous VAR {model_var_ratio[:pcnum].sum()*100:.1f}%')

# and fit model to find spon response.
analyzer = Classify_Analyzer(ac = ac,umap_model=spon_models,spon_frame=spon_series,od = 0,orien = 1,color = 0,isi = True)
analyzer.Train_SVM_Classifier(C=1)
stim_embed = analyzer.stim_embeddings
stim_label = analyzer.stim_label
spon_embed = analyzer.spon_embeddings
spon_label = analyzer.spon_label
# New operation,all shuffles.
spon_s = Spon_Shuffler(spon_series,method='phase')
spon_s_embeddings = spon_models.transform(spon_s)
spon_label_s = SVC_Fit(analyzer.svm_classifier,spon_s_embeddings,thres_prob=0)

print(f'Spon {(spon_label>0).sum()}\nShuffle {(spon_label_s>0).sum()}')


#%% ##########################  Step1, Plot PC Weight curves #########################
pcnum = 100
spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=spon_series,sample='Frame',pcnum=pcnum)

plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (12,5),dpi = 180)
sns.heatmap(spon_coords.T,center = 0,vmax = 20,vmin = -20,ax = ax,xticklabels=False,yticklabels=False)

ax.hlines([10], *ax.get_xlim(),color ='yellow')
ax.set_ylabel('PCs')
ax.set_xlabel('Frames')
ax.set_yticks([0,20,40,60,80,100])
ax.set_yticklabels([0,20,40,60,80,100])
ax.set_xticks([0,1000,2000,3000,4000,5000])
ax.set_xticklabels([0,1000,2000,3000,4000,5000])
ax.set_title(f'{pcnum} PCs Weights',size = 20)
ax.text(3700,-3,f'10 PCs explained VAR {spon_models.explained_variance_ratio_[:10].sum()*100:.1f}%')

print(f'All {pcnum} PCs explained VAR {spon_models.explained_variance_ratio_.sum()*100:.1f}%')

