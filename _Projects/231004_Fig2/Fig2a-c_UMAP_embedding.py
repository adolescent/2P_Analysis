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

work_path = r'D:\_Path_For_Figs\Fig2_UMAP_Pattern_Recognition'
expt_folder = r'D:\_All_Spon_Datas_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
ac.wp = expt_folder
# Get stim label and stim response.
all_stim_frame,all_stim_label = ac.Combine_Frame_Labels(od = True,orien = True,color = True,isi = True)
#%% Embedding spon data into umap, and save current umap model to folder. This model will be FREEZE to avoid random disruption.
reducer = ot.Load_Variable_v2(expt_folder,'All_Stim_UMAP_3D_20comp.pkl')
if reducer == False:
    kill_all_cache(r'C:\ProgramData\anaconda3\envs\umapzr')
    reducer = umap.UMAP(n_components=3,n_neighbors=20)
    reducer.fit(all_stim_frame)
    ot.Save_Variable(expt_folder,'All_Stim_UMAP_3D_20comp',reducer)
#%% Plot 3D graph of stim induced graph.
stim_embeddings = reducer.embedding_ # get spon embeddigs.
# adjust stim-frame-align train to make data better.
all_stim_labelv2 = copy.deepcopy(all_stim_label)
for i in range(len(all_stim_labelv2)-1):
    if all_stim_labelv2[i]>0 and all_stim_labelv2[i+1] == 0: # extend the end of each stim.
        all_stim_labelv2[i+1] = all_stim_labelv2[i]

#%% Plot 3d graphs.
import matplotlib.cm as cm
from mpl_toolkits import mplot3d

plt.clf()
plt.cla()

n_clusters = len(set(all_stim_label))
colors = cm.turbo(np.linspace(0, 1, n_clusters+1))# colorbars.
colors = colors[:,:3] # to solve the problem of meaningless 4D data.
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
ax.grid(False)
unique_labels = np.unique(all_stim_label)
handles = []
all_scatters = []
counter = 0
for label in unique_labels:
    mask = all_stim_label == label
    scatter = ax.scatter3D(stim_embeddings[:,0][mask], stim_embeddings[:,1][mask], stim_embeddings[:,2][mask], label=label,s = 5,facecolors = colors[counter])
    all_scatters.append(scatter)
    handles.append(scatter)
    counter +=1
# ax.legend(handles=handles,ncol = 2)
