'''
This will give V2 color recover maps. Only L85 data is possible.
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

import warnings
warnings.filterwarnings("ignore")

wp = r'D:\_All_Spon_Data_V2\L85_6B_220825'
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

spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=spon_series,sample='Frame',pcnum=pcnum)
model_var_ratio = np.array(spon_models.explained_variance_ratio_)
print(f'{pcnum} PCs explain Spontaneous VAR {model_var_ratio[:pcnum].sum()*100:.1f}%')

# and fit model to find spon response.
analyzer = UMAP_Analyzer(ac = ac,umap_model=spon_models,spon_frame=spon_series,od = 0,orien = 0,color = 1,isi = True)
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

#%% Plot PCA VARs.

plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (6,4),dpi = 144)
sns.barplot(y = model_var_ratio*100,x = np.arange(1,11),ax = ax)
ax.set_xlabel('PC',size = 12)
ax.set_ylabel('Explained Variance (%)',size = 12)
ax.set_title('Each PC explained Variance',size = 14)

#%%####################Step3, Get Recovered Graphs (Fig-6b)###############################
analyzer.Get_Stim_Spon_Compare(od = False,color = True,orien = False)
stim_graphs = analyzer.stim_recover
spon_graphs = analyzer.spon_recover
graph_lists = ['Red','Green','Blue']

plt.clf()
plt.cla()
value_max = 2
value_min = -1
font_size = 16
fig,axes = plt.subplots(nrows=2, ncols=3,figsize = (12,7),dpi = 180)
cbar_ax = fig.add_axes([.92, .35, .01, .3])

for i,c_map in enumerate(graph_lists):
    sns.heatmap(stim_graphs[c_map][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[0,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    sns.heatmap(spon_graphs[c_map][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[1,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True,cbar_kws={'label': 'Cohen D'})
    axes[0,i].set_title(c_map,size = font_size)

axes[0,0].set_ylabel('Stimulus',rotation=90,size = font_size)
axes[1,0].set_ylabel('Spontaneous',rotation=90,size = font_size)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)

analyzer.Similarity_Compare_Average(od = False,color = True,orien = False)
all_corr = analyzer.Avr_Similarity
dist = 0.26
height = 0.48
plt.figtext(0.22, height, f'R2 = {all_corr.iloc[0,0]:.3f}',size = 12)
plt.figtext(0.22+dist, height, f'R2 = {all_corr.iloc[2,0]:.3f}',size = 12)
plt.figtext(0.22+dist*2, height, f'R2 = {all_corr.iloc[4,0]:.3f}',size = 12)
cbar_ax.yaxis.label.set_size(12)
plt.show()
