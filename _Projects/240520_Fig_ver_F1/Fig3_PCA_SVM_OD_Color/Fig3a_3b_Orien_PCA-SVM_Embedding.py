'''
We will show embedding of PCA-SVM in example location.
This script show the G16 run,use as Fig 3a.
3a is combined into 3 parts: 1. Spon_embedding 2.G16 in this space 3. SVM seperated Spon embedding. The plot part can generate all 3 graphs.

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

import warnings
warnings.filterwarnings("ignore")

wp = r'D:\_All_Spon_Data_V1\L76_18M_220902'
save_path = r'D:\_Path_For_Figs\240520_Figs_ver_F1\Fig3_PCA_SVM_OD_Color'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
spon_series = ot.Load_Variable(wp,'Spon_Before.pkl')

#%%
'''
Step 0, first we need to generate the model, including svm ones
'''

spon_series = np.array(spon_series)
pcnum = 10
g16_frames,g16_labels = ac.Combine_Frame_Labels(od = 0,color = 0,orien = 1)
spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=spon_series,sample='Frame',pcnum=pcnum)
model_var_ratio = np.array(spon_models.explained_variance_ratio_)
print(f'{pcnum} PCs explain Spontaneous VAR {model_var_ratio[:pcnum].sum()*100:.1f}%')

# and fit model to find spon response.
analyzer = Classify_Analyzer(ac = ac,model=spon_models,spon_frame=spon_series,od = 0,orien = 1,color = 0,isi = True)
analyzer.Train_SVM_Classifier(C=1)
stim_embed = analyzer.stim_embeddings
stim_label = analyzer.stim_label
spon_embed = analyzer.spon_embeddings
spon_label = analyzer.spon_label

#%%
'''
Fig 3A , example of PCA embeddings, Orientation.
'''
# for convenient, plot all graphs seperately.
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
import matplotlib as mpl
import colorsys
def Plot_Colorized_Oriens(axes,embeddings,labels,pcs=[4,1,2],color_sets = np.zeros(shape = (8,3))):
    embeddings = embeddings[:,pcs]
    rest,_ = Select_Frame(embeddings,labels,used_id=[0])
    orien,orien_id = Select_Frame(embeddings,labels,used_id=list(range(9,17)))
    orien_colors = np.zeros(shape = (len(orien_id),3),dtype='f8')
    for i,c_id in enumerate(orien_id):
        orien_colors[i,:] = color_sets[int(c_id)-9,:]
    axes.scatter3D(rest[:,0],rest[:,1],rest[:,2],s = 1,c = [0.7,0.7,0.7])
    axes.scatter3D(orien[:,0],orien[:,1],orien[:,2],s = 1,c = orien_colors)
    return axes

#%% P1 Plot color bar here.
# fig,ax = plt.subplots(figsize = (2,4),dpi = 180)
color_setb = np.zeros(shape = (8,3))
fig = plt.figure(figsize = (2,4),dpi = 180)
for i,c_orien in enumerate(np.arange(0,180,22.5)):
    c_hue = c_orien/180
    c_lightness = 0.5
    c_saturation = 1
    color_setb[i,:] = colorsys.hls_to_rgb(c_hue, c_lightness, c_saturation)
cax_b = fig.add_axes([-0.5, 0, 0.08, 0.9])
custom_cmap = mcolors.ListedColormap(color_setb)
bounds = np.arange(0,202.5,22.5)
norm = mpl.colors.BoundaryNorm(bounds, custom_cmap.N)
c_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=custom_cmap),cax=cax_b, label='Orientation')
c_bar.set_ticks(np.arange(0,180,22.5)+11.25)
c_bar.set_ticklabels(np.arange(0,180,22.5))
c_bar.ax.tick_params(size=0)
#%% P2 Plot embedded 3d scatters. Change as you need.
plt.clf()
plt.cla()
plotted_pcs = [4,1,2]
u = spon_embed
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (8,4),dpi = 180,subplot_kw=dict(projection='3d'))
orien_elev = 25
orien_azim = 170
# set axes
ax.set_xlabel(f'PC {plotted_pcs[0]+1}')
ax.set_ylabel(f'PC {plotted_pcs[1]+1}')
ax.set_zlabel(f'PC {plotted_pcs[2]+1}')
ax.grid(False)
ax.view_init(elev=orien_elev, azim=orien_azim)
ax.set_box_aspect(aspect=None, zoom=0.85) # shrink graphs
ax.axes.set_xlim3d(left=-20, right=30)
ax.axes.set_ylim3d(bottom=-30, top=40)
ax.axes.set_zlim3d(bottom=20, top=-20)
# tmp_planes = ax.zaxis._PLANES 
# ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
#                         tmp_planes[0], tmp_planes[1], 
#                         tmp_planes[4], tmp_planes[5])
# ax = Plot_Colorized_Oriens(ax,spon_embed,np.zeros(len(spon_embed)),plotted_pcs,color_setb)
ax = Plot_Colorized_Oriens(ax,stim_embed,stim_label,plotted_pcs,color_setb)
# ax = Plot_Colorized_Oriens(ax,spon_embed,spon_label,plotted_pcs,color_setb)
# ax = Plot_Colorized_Oriens(ax,spon_s_embeddings,spon_label_s,plotted_pcs,color_setb)
ax.set_title('Classified Spontaneous in PCA Space',size = 10)
# ax.set_title('Orientation Stimulus in PCA Space',size = 10)
# ax.set_title('Shuffled Spontaneous in PCA Space',size = 10)

fig.tight_layout()

#%%
'''
Fig 3B, we get recovered graph and compare it with real stim graph
'''

analyzer.Get_Stim_Spon_Compare(od = False,color = False)
stim_graphs = analyzer.stim_recover
spon_graphs = analyzer.spon_recover
graph_lists = ['Orien0','Orien45','Orien90','Orien135']
analyzer.Similarity_Compare_Average(od = False,color = False)
all_corr = analyzer.Avr_Similarity

plt.clf()
plt.cla()
value_max = 2
value_min = -1
font_size = 16
fig,axes = plt.subplots(nrows=2, ncols=4,figsize = (14,7),dpi = 180)
cbar_ax = fig.add_axes([.92, .45, .01, .2])

for i,c_map in enumerate(graph_lists):
    sns.heatmap(stim_graphs[c_map][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[1,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    sns.heatmap(spon_graphs[c_map][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[0,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True,cbar_kws={'label': 'Z Scored Activity'})
    axes[0,i].set_title(c_map,size = font_size)

axes[1,0].set_ylabel('Stimulus',rotation=90,size = font_size)
axes[0,0].set_ylabel('Spontaneous',rotation=90,size = font_size)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
dist = 0.195
height = 0.485
plt.figtext(0.18, height, f'R2 = {all_corr.iloc[0,0]:.3f}',size = 14)
plt.figtext(0.18+dist, height, f'R2 = {all_corr.iloc[2,0]:.3f}',size = 14)
plt.figtext(0.18+dist*2, height, f'R2 = {all_corr.iloc[4,0]:.3f}',size = 14)
plt.figtext(0.18+dist*3, height, f'R2 = {all_corr.iloc[6,0]:.3f}',size = 14)
cbar_ax.yaxis.label.set_size(12)
# fig.tight_layout()

plt.show()