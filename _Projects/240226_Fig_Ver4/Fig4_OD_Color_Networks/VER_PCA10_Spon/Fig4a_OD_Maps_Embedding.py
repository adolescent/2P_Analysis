'''
This graph will generate embedding of OD maps in PCA space. The same method as we do on Orietation PCAs.

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

example_loc = r'D:\_All_Spon_Data_V1\L76_18M_220902'
ac = ot.Load_Variable(example_loc,'Cell_Class.pkl')
spon_series = ot.Load_Variable(example_loc,'Spon_Before.pkl')
# if we need raw frame dF values
# raw_orien_run = ot.Load_Variable(f'{example_loc}\\Orien_Frames_Raw.pkl')
# raw_spon_run = ot.Load_Variable(f'{example_loc}\\Spon_Before_Raw.pkl')

wp = r'D:\_Path_For_Figs\240228_Figs_v4\Fig4'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)


#%% ########################### Step0, Calculate PCA Model (OD)########################################

pcnum = 10
spon_series = np.array(spon_series)
od_frames,od_labels = ac.Combine_Frame_Labels(od = 1,color = 0,orien = 0)
spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=spon_series,sample='Frame',pcnum=pcnum)
model_var_ratio = np.array(spon_models.explained_variance_ratio_)
print(f'{pcnum} PCs explain Spontaneous VAR {model_var_ratio[:pcnum].sum()*100:.1f}%')

# and fit model to find spon response.
analyzer = Classify_Analyzer(ac = ac,umap_model=spon_models,spon_frame=spon_series,od = 1,orien = 0,color = 0,isi = True)
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

#%%####################### Step1, Plot 3 embedding maps - OD (Fig 4A)##################################

import matplotlib.cm as cm
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
import matplotlib as mpl
import colorsys

plotted_pcs = [2,3,5]
orien_elev = 15
orien_azim = 150
zoom = 1

fig,ax = plt.subplots(nrows=3, ncols=1,figsize = (12,10),dpi = 180,subplot_kw=dict(projection='3d'))

# Grid Preparing
for i in range(3):
    ax[i].set_xlabel(f'PC {plotted_pcs[0]+1}')
    ax[i].set_ylabel(f'PC {plotted_pcs[1]+1}')
    ax[i].set_zlabel(f'PC {plotted_pcs[2]+1}')
    ax[i].grid(False)
    ax[i].view_init(elev=orien_elev, azim=orien_azim)
    ax[i].set_box_aspect(aspect=None, zoom=1) # shrink graphs
    ax[i].axes.set_xlim3d(left=-15, right=30)
    ax[i].axes.set_ylim3d(bottom=-25, top=25)
    ax[i].axes.set_zlim3d(bottom=-20, top=20)
    # ax[i].set_position([ax[i].get_position().x0-0.12, ax[i].get_position().y0, ax[i].get_position().width*zoom, ax[i].get_position().height*zoom])
    # set z label location
    tmp_planes = ax[i].zaxis._PLANES 
    ax[i].zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                            tmp_planes[0], tmp_planes[1], 
                            tmp_planes[4], tmp_planes[5])

## get OD color bars.
color_setb = np.array([[1,0,0],[0,1,0]])
cax_b = fig.add_axes([0.3, 0.4, 0.02, 0.3])
custom_cmap = mcolors.ListedColormap(color_setb)
bounds = np.arange(0,3,1)
norm = mpl.colors.BoundaryNorm(bounds, custom_cmap.N)
c_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=custom_cmap),cax=cax_b, label='Best Eye')
c_bar.set_ticks(np.arange(0,2,1)+0.5)
c_bar.set_ticklabels(['LE','RE'])
c_bar.ax.tick_params(size=0)

# plot colorized graphs
def Plot_Colorized_OD(axes,embeddings,labels,pcs=plotted_pcs,color_sets = color_setb):
    embeddings = embeddings[:,pcs]
    rest,_ = Select_Frame(embeddings,labels,used_id=[0])
    od,od_ids = Select_Frame(embeddings,labels,used_id=list(range(1,9)))
    od_colors = np.zeros(shape = (len(od_ids),3),dtype='f8')
    for i,c_id in enumerate(od_ids):
        od_colors[i,:] = color_sets[int(c_id)%2,:]
    axes.scatter3D(rest[:,0],rest[:,1],rest[:,2],s = 1,c = [0.7,0.7,0.7],alpha = 0.1)
    axes.scatter3D(od[:,0],od[:,1],od[:,2],s = 1,c = od_colors)
    return axes

ax[0] = Plot_Colorized_OD(ax[0],stim_embed,stim_label,plotted_pcs,color_setb)
ax[1] = Plot_Colorized_OD(ax[1],spon_embed,spon_label,plotted_pcs,color_setb)
ax[2] = Plot_Colorized_OD(ax[2],spon_s_embeddings,spon_label_s,plotted_pcs,color_setb)

# set title
ax[0].set_title('Stimulus Embedding in PCA Space',size = 10)
ax[1].set_title('Spontaneous Embedding in PCA Space',size = 10)
ax[2].set_title('Shuffled Phase Embedding in PCA Space',size = 10)

# fig.tight_layout()

#%%####################Step3, Get Recovered Graphs (Fig-4b)###############################

analyzer.Get_Stim_Spon_Compare(od = True,color = False,orien = False)
stim_graphs = analyzer.stim_recover
spon_graphs = analyzer.spon_recover
graph_lists = ['LE','RE']

plt.clf()
plt.cla()
value_max = 2
value_min = -1
font_size = 16
fig,axes = plt.subplots(nrows=2, ncols=2,figsize = (6,6),dpi = 180)
cbar_ax = fig.add_axes([.99, .15, .02, .7])
for i,c_map in enumerate(graph_lists):
    sns.heatmap(stim_graphs[c_map][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[0,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    sns.heatmap(spon_graphs[c_map][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[1,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    axes[0,i].set_title(c_map,size = font_size)

axes[0,0].set_ylabel('Stimulus',rotation=90,size = font_size)
axes[1,0].set_ylabel('Spontaneous',rotation=90,size = font_size)

#%% ##################Get Recovered Graphs (Fig-4b) -- ver frame##########################
raw_od_run = ot.Load_Variable(f'{example_loc}\\OD_Frames_Raw.pkl')
raw_spon_run = ot.Load_Variable(f'{example_loc}\\Spon_Before_Raw.pkl')
orien_avr = raw_od_run.mean(0)
spon_avr = raw_spon_run.mean(0)

spon_label_od = np.zeros(len(spon_label))
for i in range(len(spon_label)):
    if (spon_label[i]>0) and (spon_label[i]%2 == 1):
        spon_label_od[i] = 1
    elif (spon_label[i]>0):
        spon_label_od[i] = 2
    else:
        spon_label_od[i] = 0
stim_label_od = np.zeros(len(stim_label))
for i in range(len(stim_label)):
    if (stim_label[i]>0) and (stim_label[i]%2 == 1):
        stim_label_od[i] = 1
    elif (stim_label[i]>0):
        stim_label_od[i] = 2
    else:
        stim_label_od[i] = 0


graph_lists = ['LE','RE']
graph_ids = [1,2]
all_stim_maps = {}
all_spon_recover_maps = {}
clip_std = 5
# egt all avr stim & spon maps
for i,c_map in tqdm(enumerate(graph_lists)):
    c_spons = raw_spon_run[np.where(spon_label_od == graph_ids[i])[0],:,:]
    c_spon_avr = c_spons.mean(0)-spon_avr
    c_spon_avr = np.clip(c_spon_avr,(c_spon_avr.mean()-clip_std*c_spon_avr.std()),(c_spon_avr.mean()+clip_std*c_spon_avr.std()))
    c_stims = raw_od_run[np.where(stim_label_od == graph_ids[i])[0],:,:]
    c_stim_avr = c_stims.mean(0)-orien_avr
    c_stim_avr = np.clip(c_stim_avr,(c_stim_avr.mean()-clip_std*c_stim_avr.std()),(c_stim_avr.mean()+clip_std*c_stim_avr.std()))
    all_stim_maps[c_map] = c_stim_avr 
    all_spon_recover_maps[c_map] = c_spon_avr

#%% Plot Graph Compare Maps
frame_corrs = []
plt.clf()
plt.cla()
value_max = 300
value_min = -300
font_size = 16
fig,axes = plt.subplots(nrows=2, ncols=2,figsize = (6,6),dpi = 180)
cbar_ax = fig.add_axes([.99, .15, .02, .7])
for i,c_map in enumerate(graph_lists):
    plotable_c_stim = all_stim_maps[c_map]
    # plotable_c_stim = plotable_c_stim/plotable_c_stim.std()
    sns.heatmap(plotable_c_stim,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)

    plotable_c_spon = all_spon_recover_maps[c_map]
    # plotable_c_spon = plotable_c_spon/plotable_c_spon.std()
    c_corr,_ = stats.pearsonr(all_spon_recover_maps[c_map][20:492,20:492].flatten(),all_stim_maps[c_map][20:492,20:492].flatten())
    frame_corrs.append(c_corr)
    sns.heatmap(plotable_c_spon,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    axes[0,i].set_title(c_map,size = font_size)

axes[0,0].set_ylabel('Stimulus',rotation=90,size = font_size)
axes[1,0].set_ylabel('Spontaneous',rotation=90,size = font_size)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
fig.tight_layout()
plt.show()

