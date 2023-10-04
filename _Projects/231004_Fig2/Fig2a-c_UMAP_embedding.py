'''
This script will do umap training and embedding on example data. All will be redo so be careful.
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

work_path = r'D:\_Path_For_Figs\Fig2_UMAP_Pattern_Recognition'
expt_folder = r'D:\_All_Spon_Datas_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
ac.wp = expt_folder
# Get stim label and stim response.
all_stim_frame,all_stim_label = ac.Combine_Frame_Labels(od = True,orien = True,color = True,isi = True)
spon_series = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
#%% Embedding spon data into umap, and save current umap model to folder. This model will be FREEZE to avoid random disruption.
reducer = ot.Load_Variable_v2(expt_folder,'All_Stim_UMAP_3D_20comp.pkl')
if reducer == False:
    kill_all_cache(r'C:\ProgramData\anaconda3\envs\umapzr')
    reducer = umap.UMAP(n_components=3,n_neighbors=20)
    reducer.fit(all_stim_frame)
    ot.Save_Variable(expt_folder,'All_Stim_UMAP_3D_20comp',reducer)
#%% Get embeddings and adjust label.
stim_embeddings = reducer.embedding_ # get spon embeddigs.
# adjust stim-frame-align train to make data better.
all_stim_labelv2 = copy.deepcopy(all_stim_label)
jump_next = 0
for i in range(len(all_stim_labelv2)-1):
    if jump_next == 1:
        jump_next = 0
        continue
    if all_stim_labelv2[i]>0 and all_stim_labelv2[i+1] == 0: # extend the end of each stim.
        all_stim_labelv2[i+1] = all_stim_labelv2[i]
        jump_next = 1
#%% Embedding spontaneous data, and perform SVM classification.
spon_embeddings = reducer.transform(spon_series)
# and we tran an SVM based on stim data
classifier,score = SVM_Classifier(embeddings=stim_embeddings,label = all_stim_labelv2)
predicted_spon_label = SVC_Fit(classifier,data = spon_embeddings,thres_prob = 0)

#%%###############################################################################
# Fig 2b, Plot 3d graphs. Stim and spon in the same page.
import matplotlib.cm as cm
from mpl_toolkits import mplot3d

plt.clf()
plt.cla()
# set graph
fig,axes = plt.subplots(nrows=1, ncols=2,figsize = (16,7),dpi = 180,subplot_kw=dict(projection='3d'))
elev = 25 # up-down angle
azim = 30 # rotation angle
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
n_clusters = len(set(all_stim_labelv2))
colors = cm.turbo(np.linspace(0, 1, n_clusters+1))[:,:3]
# Plot Stim graph on Plotter 1.
unique_labels = np.unique(all_stim_labelv2)
handles = [] # use for label.
all_scatters = []
counter = 0
for label in unique_labels:
    mask = all_stim_labelv2 == label
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
axes[0].axes.set_xlim3d(left=2, right=14) 
axes[0].axes.set_ylim3d(bottom=2, top=12) 
axes[0].axes.set_zlim3d(bottom=5, top=15) 
axes[1].axes.set_xlim3d(left=2, right=14) 
axes[1].axes.set_ylim3d(bottom=2, top=12) 
axes[1].axes.set_zlim3d(bottom=5, top=15) 
axes[0].set_xlabel('UMAP 1')
axes[0].set_ylabel('UMAP 2')
axes[0].set_zlabel('UMAP 3')
axes[1].set_xlabel('UMAP 1')
axes[1].set_ylabel('UMAP 2')
axes[1].set_zlabel('UMAP 3')
axes[0].set_title('Stimulus Response in UMAP Space',size = 24)
axes[1].set_title('Spontaneous Response in UMAP Space',size = 24)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
plt.tight_layout(pad = 6)
plt.show()
#%%###############################################################################
# Fig 2c,Compare stim graph with spon graph.
# get 6 stimulus functional reaction.
LE_map = ac.OD_t_graphs['L-0'].loc['A_reponse']
RE_map = ac.OD_t_graphs['R-0'].loc['A_reponse']
Orien0_map = ac.Orien_t_graphs['Orien0-0'].loc['A_reponse']
Orien45_map = ac.Orien_t_graphs['Orien45-0'].loc['A_reponse']
Orien90_map = ac.Orien_t_graphs['Orien90-0'].loc['A_reponse']
Orien135_map = ac.Orien_t_graphs['Orien135-0'].loc['A_reponse']
# get spon recovered maps.
LE_ids = np.where((predicted_spon_label>0)*(predicted_spon_label<9)*(predicted_spon_label%2==1))[0]
LE_recovered_map = spon_series.iloc[list(LE_ids),:].mean(0)
RE_ids = np.where((predicted_spon_label>0)*(predicted_spon_label<9)*(predicted_spon_label%2==0))[0]
RE_recovered_map = spon_series.iloc[list(RE_ids),:].mean(0)
Orien0_ids = np.where((predicted_spon_label==9))[0]
Orien0_recovered_map = spon_series.iloc[list(Orien0_ids ),:].mean(0)
Orien45_ids = np.where((predicted_spon_label==11))[0]
Orien45_recovered_map = spon_series.iloc[list(Orien45_ids ),:].mean(0)
Orien90_ids = np.where((predicted_spon_label==13))[0]
Orien90_recovered_map = spon_series.iloc[list(Orien90_ids ),:].mean(0)
Orien135_ids = np.where((predicted_spon_label==15))[0]
Orien135_recovered_map = spon_series.iloc[list(Orien135_ids ),:].mean(0)
# Pad stim and spon graph,
LE_compare = np.hstack((ac.Generate_Weighted_Cell(LE_map),ac.Generate_Weighted_Cell(LE_recovered_map)))
RE_compare = np.hstack((ac.Generate_Weighted_Cell(RE_map),ac.Generate_Weighted_Cell(RE_recovered_map)))
Orien0_compare = np.hstack((ac.Generate_Weighted_Cell(Orien0_map),ac.Generate_Weighted_Cell(Orien0_recovered_map)))
Orien45_compare = np.hstack((ac.Generate_Weighted_Cell(Orien45_map),ac.Generate_Weighted_Cell(Orien45_recovered_map)))
Orien90_compare = np.hstack((ac.Generate_Weighted_Cell(Orien90_map),ac.Generate_Weighted_Cell(Orien90_recovered_map)))
Orien135_compare = np.hstack((ac.Generate_Weighted_Cell(Orien135_map),ac.Generate_Weighted_Cell(Orien135_recovered_map)))
# pad data boulder.
LE_compare[:,510:514] = 10
RE_compare[:,510:514] = 10
Orien0_compare[:,510:514] = 10
Orien45_compare[:,510:514] = 10
Orien90_compare[:,510:514] = 10
Orien135_compare[:,510:514] = 10
#%% plot results in subplots
value_max = 2
value_min = -1
font_size = 11
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14,5),dpi = 180)

cbar_ax = fig.add_axes([.99, .15, .02, .7])
sns.heatmap(LE_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,0],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(RE_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,0],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(Orien0_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,1],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(Orien45_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,2],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(Orien90_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,1],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(Orien135_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,2],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)

axes[0,0].set_title('Left Eye Stimulus                Left Eye Recovered',size = font_size)
axes[1,0].set_title('Right Eye Stimulus              Right Eye Recovered',size = font_size)
axes[0,1].set_title('Orientation0 Stimulus        Orientation0 Recovered',size = font_size)
axes[0,2].set_title('Orientation45 Stimulus      Orientation45 Recovered',size = font_size)
axes[1,1].set_title('Orientation90 Stimulus      Orientation90 Recovered',size = font_size)
axes[1,2].set_title('Orientation135 Stimulus    Orientation135 Recovered',size = font_size)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=None)
fig.tight_layout()
plt.show()
# ac.Generate_Weighted_Cell(LE_recovered_map)
# axes.set_aspect(aspect = 1)