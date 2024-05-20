'''
This is the first step of Fig2 Ver. A.

This Ver. graph do pca on spon graph directly, first we need to determine the useful PC num.

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
raw_orien_run = ot.Load_Variable(f'{wp}\\Orien_Frames_Raw.pkl')
raw_spon_run = ot.Load_Variable(f'{wp}\\Spon_Before_Raw.pkl')

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
#%%####################### Step1, Plot 3 embedding maps (Fig 2A)##################################
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
import matplotlib as mpl
import colorsys

plotted_pcs = [4,1,2]
orien_elev = 25
orien_azim = 170

fig,ax = plt.subplots(nrows=3, ncols=1,figsize = (8,12),dpi = 180,subplot_kw=dict(projection='3d'))

# Grid Preparing
for i in range(3):
    ax[i].set_xlabel(f'PC {plotted_pcs[0]+1}')
    ax[i].set_ylabel(f'PC {plotted_pcs[1]+1}')
    ax[i].set_zlabel(f'PC {plotted_pcs[2]+1}')
    ax[i].grid(False)
    ax[i].view_init(elev=orien_elev, azim=orien_azim)
    ax[i].set_box_aspect(aspect=None, zoom=0.87) # shrink graphs
    ax[i].axes.set_xlim3d(left=-20, right=30)
    ax[i].axes.set_ylim3d(bottom=-30, top=40)
    ax[i].axes.set_zlim3d(bottom=20, top=-20)

## get orien color bars.
color_setb = np.zeros(shape = (8,3))
for i,c_orien in enumerate(np.arange(0,180,22.5)):
    c_hue = c_orien/180
    c_lightness = 0.5
    c_saturation = 1
    color_setb[i,:] = colorsys.hls_to_rgb(c_hue, c_lightness, c_saturation)
cax_b = fig.add_axes([0.15, 0.4, 0.02, 0.3])
custom_cmap = mcolors.ListedColormap(color_setb)
bounds = np.arange(0,202.5,22.5)
norm = mpl.colors.BoundaryNorm(bounds, custom_cmap.N)
c_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=custom_cmap),cax=cax_b, label='Orientation')
c_bar.set_ticks(np.arange(0,180,22.5)+11.25)
c_bar.set_ticklabels(np.arange(0,180,22.5))
c_bar.ax.tick_params(size=0)

# plot colorized graphs
def Plot_Colorized_Oriens(axes,embeddings,labels,pcs=plotted_pcs,color_sets = color_setb):
    embeddings = embeddings[:,pcs]
    rest,_ = Select_Frame(embeddings,labels,used_id=[0])
    orien,orien_id = Select_Frame(embeddings,labels,used_id=list(range(9,17)))
    orien_colors = np.zeros(shape = (len(orien_id),3),dtype='f8')
    for i,c_id in enumerate(orien_id):
        orien_colors[i,:] = color_sets[int(c_id)-9,:]
    axes.scatter3D(rest[:,0],rest[:,1],rest[:,2],s = 1,c = [0.7,0.7,0.7],alpha = 0.1)
    axes.scatter3D(orien[:,0],orien[:,1],orien[:,2],s = 1,c = orien_colors)
    return axes

ax[0] = Plot_Colorized_Oriens(ax[0],stim_embed,stim_label,plotted_pcs,color_setb)
ax[1] = Plot_Colorized_Oriens(ax[1],spon_embed,spon_label,plotted_pcs,color_setb)
ax[2] = Plot_Colorized_Oriens(ax[2],spon_s_embeddings,spon_label_s,plotted_pcs,color_setb)

# set title
ax[0].set_title('Stimulus Embedding in PCA Space',size = 10)
ax[1].set_title('Spontaneous Embedding in PCA Space',size = 10)
ax[2].set_title('Shuffled Phase Embedding in PCA Space',size = 10)

fig.tight_layout()

#%%####################Step3, Get Recovered Graphs (Fig-2b)###############################
analyzer.Get_Stim_Spon_Compare(od = False,color = False)
stim_graphs = analyzer.stim_recover
spon_graphs = analyzer.spon_recover
graph_lists = ['Orien0','Orien45','Orien90','Orien135']
plt.clf()
plt.cla()
value_max = 2
value_min = -1
font_size = 16
fig,axes = plt.subplots(nrows=2, ncols=4,figsize = (14,6),dpi = 180)
cbar_ax = fig.add_axes([.99, .15, .02, .7])

for i,c_map in enumerate(graph_lists):
    sns.heatmap(stim_graphs[c_map][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[0,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    sns.heatmap(spon_graphs[c_map][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[1,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    axes[0,i].set_title(c_map,size = font_size)

axes[0,0].set_ylabel('Stimulus',rotation=90,size = font_size)
axes[1,0].set_ylabel('Spontaneous',rotation=90,size = font_size)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
fig.tight_layout()
plt.show()
analyzer.Similarity_Compare_Average(od = False,color = False)
all_corr = analyzer.Avr_Similarity
#%% ##################Get Recovered Graphs (Fig-2b) -- ver frame##########################
orien_avr = raw_orien_run.mean(0)
spon_avr = raw_spon_run.mean(0)
graph_lists = ['Orien0','Orien45','Orien90','Orien135']
graph_ids = [9,11,13,15]
all_stim_maps = {}
all_spon_recover_maps = {}
clip_std = 5
# egt all avr stim & spon maps
for i,c_map in tqdm(enumerate(graph_lists)):
    c_spons = raw_spon_run[np.where(spon_label == graph_ids[i])[0],:,:]
    c_spon_avr = c_spons.mean(0)-spon_avr
    c_spon_avr = np.clip(c_spon_avr,(c_spon_avr.mean()-clip_std*c_spon_avr.std()),(c_spon_avr.mean()+clip_std*c_spon_avr.std()))
    c_stims = raw_orien_run[np.where(stim_label == graph_ids[i])[0],:,:]
    c_stim_avr = c_stims.mean(0)-orien_avr
    c_stim_avr = np.clip(c_stim_avr,(c_stim_avr.mean()-clip_std*c_stim_avr.std()),(c_stim_avr.mean()+clip_std*c_stim_avr.std()))
    all_stim_maps[c_map] = c_stim_avr 
    all_spon_recover_maps[c_map] = c_spon_avr
#%% Plot Graph Compare Maps
frame_corrs = []
plt.clf()
plt.cla()
value_max = 4
value_min = -4
font_size = 16
fig,axes = plt.subplots(nrows=2, ncols=4,figsize = (14,6),dpi = 180)
cbar_ax = fig.add_axes([.99, .15, .02, .7])

for i,c_map in enumerate(graph_lists):
    plotable_c_stim = all_stim_maps[c_map]
    plotable_c_stim = plotable_c_stim/plotable_c_stim.std()
    sns.heatmap(plotable_c_stim,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)

    plotable_c_spon = all_spon_recover_maps[c_map]
    plotable_c_spon = plotable_c_spon/plotable_c_spon.std()
    c_corr,_ = stats.pearsonr(all_spon_recover_maps[c_map][20:492,20:492].flatten(),all_stim_maps[c_map][20:492,20:492].flatten())
    frame_corrs.append(c_corr)
    sns.heatmap(plotable_c_spon,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    axes[0,i].set_title(c_map,size = font_size)

axes[0,0].set_ylabel('Stimulus',rotation=90,size = font_size)
axes[1,0].set_ylabel('Spontaneous',rotation=90,size = font_size)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
fig.tight_layout()
plt.show()


#%% ################### Frame 2b ver 3, Similarity Compare ################################
# METHOD DISCARD, NO GOOD RESULTS.
## generate all cell resposne matrix
# graph_lists = ['Orien0','Orien45','Orien90','Orien135']
# all_cell_info = pd.DataFrame(columns = ['Z_value','Data','Map','Cell'])

# for i,c_map in tqdm(enumerate(graph_lists)):
#     c_stim_map = analyzer.stim_recover[c_map][0]
#     c_spon_map = analyzer.spon_recover[c_map][0]
#     for j in range(len(c_stim_map)):
#         all_cell_info.loc[len(all_cell_info),:] = [c_stim_map[j],'Stim',c_map,j+1]
#         all_cell_info.loc[len(all_cell_info),:] = [c_spon_map[j],'Spon',c_map,j+1]
        
# #%% Plot similarity barplots.
# c_map = all_cell_info.groupby('Map').get_group('Orien45')

# sorted_cell_sequence = c_map.groupby('Data').get_group('Stim').sort_values(by=['Z_value'],ascending=False).index


# plt.clf()
# plt.cla()
# fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (20,4),dpi = 180)
# sns.barplot(data = c_map,x = 'Cell',y = 'Z_value',hue = 'Data',ax = ax)
# ax.set(xticklabels=[])
# ax.set(xticks=[])

#%% ########################### Fig 2C - PC Axis  ######################
all_pc_graphs = np.zeros(shape = (512,512,10),dtype='f8')
corr_oriens = pd.DataFrame(columns = ['Orien0','Orien45','Orien90','Orien135','LE','RE','Red','Green','Blue'])

stim_frames,stim_labels = ac.Combine_Frame_Labels(od = 1,orien = 1,color = 1)
for i in range(spon_pcs.shape[0]):
    c_response = spon_pcs[i,:]
    c_map = ac.Generate_Weighted_Cell(c_response)
    all_pc_graphs[:,:,i] = c_map
    o0_map = Select_Frame(stim_frames,stim_labels,[9])[0].mean(0)
    o45_map = Select_Frame(stim_frames,stim_labels,[11])[0].mean(0)
    o90_map = Select_Frame(stim_frames,stim_labels,[13])[0].mean(0)
    o135_map = Select_Frame(stim_frames,stim_labels,[15])[0].mean(0)
    LE_map = Select_Frame(stim_frames,stim_labels,[1,3,5,7])[0].mean(0)
    RE_map = Select_Frame(stim_frames,stim_labels,[2,4,6,8])[0].mean(0)
    Red_map = Select_Frame(stim_frames,stim_labels,[17])[0].mean(0)
    Green_map = Select_Frame(stim_frames,stim_labels,[19])[0].mean(0)
    Blue_map = Select_Frame(stim_frames,stim_labels,[21])[0].mean(0)

    o0_corr,_ = stats.pearsonr(c_response,o0_map)
    o45_corr,_ = stats.pearsonr(c_response,o45_map)
    o90_corr,_ = stats.pearsonr(c_response,o90_map)
    o135_corr,_ = stats.pearsonr(c_response,o135_map)
    le_corr,_ = stats.pearsonr(c_response,LE_map)
    re_corr,_ = stats.pearsonr(c_response,RE_map)
    red_corr,_ = stats.pearsonr(c_response,Red_map)
    green_corr,_ = stats.pearsonr(c_response,Green_map)
    blue_corr,_ = stats.pearsonr(c_response,Blue_map)

    corr_oriens.loc[len(corr_oriens),:] = [o0_corr,o45_corr,o90_corr,o135_corr,le_corr,re_corr,red_corr,green_corr,blue_corr]
    
#%% Plot 10 PCs 
plt.clf()
plt.cla()

vmax = 0.12
vmin = -0.12
fig,axes = plt.subplots(nrows=2, ncols=5,figsize = (10,5.5),dpi = 180)
cbar_ax = fig.add_axes([1, .15, .02, .7])

for i in range(spon_pcs.shape[0]):
    c_map = all_pc_graphs[:,:,i]
    sns.heatmap(c_map,ax = axes[i//5,i%5],center = 0,xticklabels=False,yticklabels=False,cbar_ax= cbar_ax,square=True,vmax = vmax,vmin = vmin)
    axes[i//5,i%5].set_title(f'PC {i+1} ')

fig.tight_layout()

#%% Plot PC's corr to Stim Maps.
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (10,8),dpi = 180)
data_for_plot = abs(corr_oriens.astype('f8'))
font_size = 20
cbar_ax = fig.add_axes([1, .15, .02, .7])
sns.heatmap(data_for_plot,annot = True,fmt = '.2f',center = 0,ax = ax,cbar_ax=cbar_ax,vmax = 0.6)
ax.set_title('PCs Correlation with Functional Maps',size = font_size)
ax.set_ylabel('PC Number',size = font_size-2)
ax.set_yticklabels(range(1,11))
ax.set_xlabel('Functional Maps',size = font_size-2)
ax.tick_params(axis='both', which='major', labelsize=14)
fig.tight_layout()
