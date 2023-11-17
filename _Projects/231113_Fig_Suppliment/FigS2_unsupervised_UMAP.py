'''
This script will generate unsupervised umap result to compare with supervised ones.
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
from Filters import Signal_Filter

work_path = r'D:\_Path_For_Figs\Fig2_UMAP_Pattern_Recognition'
expt_folder = r'D:\_All_Spon_Datas_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
ac.wp = expt_folder
# Get stim label and stim response.
all_stim_frame,all_stim_label = ac.Combine_Frame_Labels(od = True,orien = True,color = True,isi = True)
spon_series = ot.Load_Variable(expt_folder,'Spon_Before.pkl')


#%%########################## UNSUPER REDUCER ########################### 
# Step1, get reducer
reducer_unsu = umap.UMAP(n_components=3,n_neighbors=20)
reducer_unsu.fit(spon_series)
#%% Step2, embedding stim into spon space.
stim_embeddings = reducer_unsu.transform(all_stim_frame)
spon_embeddings = reducer_unsu.embedding_
#%% Step3, Classify Stim using UMAP-SVM
classifier,score = SVM_Classifier(embeddings=stim_embeddings,label = all_stim_label,C = 10)
predicted_spon_label = SVC_Fit(classifier,data = spon_embeddings,thres_prob = 0)
#%%#################### PLOT UNSUPERVISED RESULT ###################################
#%% 3D Plotter
import matplotlib.cm as cm
from mpl_toolkits import mplot3d

plt.clf()
plt.cla()
# set graph
fig,axes = plt.subplots(nrows=1, ncols=2,figsize = (16,7),dpi = 180,subplot_kw=dict(projection='3d'))
elev = 25 # up-down angle
azim = 330 # rotation angle
axes[0].grid(False)
axes[1].grid(False)
axes[0].view_init(elev=elev, azim=azim)
axes[1].view_init(elev=elev, azim=azim)
#### Set ticks using switch below.
# ax.set_xticks([0,2,4,6,8,10])
# ax.set_yticks([0,2,4,6,8,10])
# ax.set_zticks([0,2,4,6,8,10])
# ax.axes.set_xlim3d(left=0.2, right=9.8) 
# ax.axes.set_ylim3d(bottom=0.2, top=9.8) 
# ax.axes.set_zlim3d(bottom=0.2, top=9.8) 
n_clusters = len(set(all_stim_label))
colors = cm.turbo(np.linspace(0, 1, n_clusters+1))[:,:3]
# Plot Stim graph on Plotter 1.
unique_labels = np.unique(all_stim_label)
handles = [] # use for label.
all_scatters = []
counter = 0
for label in unique_labels:
    mask = all_stim_label == label
    scatter = axes[0].scatter3D(stim_embeddings[:,0][mask], stim_embeddings[:,1][mask], stim_embeddings[:,2][mask], label=label,s = 5,facecolors = colors[counter])
    all_scatters.append(scatter)
    handles.append(scatter)
    counter +=1
# ax.legend(handles=handles,ncol = 2) # if you need legend.
## Plot Spon SVM trained graph.
handles_spon = [] # use for label.
all_scatters_spon = []
counter = 0
for label in unique_labels:
    mask = predicted_spon_label == label
    scatter = axes[1].scatter3D(spon_embeddings[:,0][mask], spon_embeddings[:,1][mask], spon_embeddings[:,2][mask], label=label,s = 5,facecolors = colors[counter])
    all_scatters_spon.append(scatter)
    handles_spon.append(scatter)
    counter +=1
#### Adjustment and Labels of graphs. With small fix.
axes[0].axes.set_xlim3d(left=4, right=9) 
axes[0].axes.set_ylim3d(bottom=0, top=9) 
axes[0].axes.set_zlim3d(bottom=4, top=10) 
axes[1].axes.set_xlim3d(left=4, right=9) 
axes[1].axes.set_ylim3d(bottom=0, top=9) 
axes[1].axes.set_zlim3d(bottom=4, top=10) 
axes[0].set_xlabel('UMAP 1')
axes[0].set_ylabel('UMAP 2')
axes[0].set_zlabel('UMAP 3')
axes[1].set_xlabel('UMAP 1')
axes[1].set_ylabel('UMAP 2')
axes[1].set_zlabel('UMAP 3')
axes[0].set_title('Stimulus Response in UMAP Space',size = 20)
axes[1].set_title('Spontaneous Response in UMAP Space',size = 20)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
plt.tight_layout(pad = 6)
plt.show()

#%%######################## PLOT RECOVERED MAP ############################### 
# 1. Generate recovered map
RE_locs = np.where((predicted_spon_label == 6)+(predicted_spon_label == 8))[0]
RE_recover = spon_series.iloc[RE_locs,:].mean(0)
RE_map = ac.OD_t_graphs['R-0'].loc['A_reponse']

Orien90_locs = np.where(predicted_spon_label == 13)[0]
Orien90_recover = spon_series.iloc[Orien90_locs,:].mean(0)
Orien90_map = ac.Orien_t_graphs['Orien90-0'].loc['A_reponse']

Red_locs = np.where(predicted_spon_label == 17)[0]
Red_recover = spon_series.iloc[Red_locs,:].mean(0)
Red_map = ac.Color_t_graphs['Red-0'].loc['A_reponse']

RE_compare = np.hstack((ac.Generate_Weighted_Cell(RE_map),ac.Generate_Weighted_Cell(RE_recover)))
Orien90_compare = np.hstack((ac.Generate_Weighted_Cell(Orien90_map),ac.Generate_Weighted_Cell(Orien90_recover)))
Red_compare = np.hstack((ac.Generate_Weighted_Cell(Red_map),ac.Generate_Weighted_Cell(Red_recover)))

RE_compare[:,510:514] = 10
Orien90_compare[:,510:514] = 10
Red_compare[:,510:514] = 10
#%% 2.Plot recovered map
plt.clf()
plt.cla()
value_max = 3
value_min = -1
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(4,6),dpi = 180)
cbar_ax = fig.add_axes([0.95, .35, .03, .3])
sns.heatmap(RE_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(Orien90_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(Red_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[2],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
axes[0].set_title('Right Eye Stimulus              Right Eye Recovered',size = 8)
axes[1].set_title('Orientation 90 Stimulus        Orientation 90 Recovered',size = 8)
axes[2].set_title('Red Stimulus                        Red Recovered',size = 8)
fig.tight_layout()
plt.show()

