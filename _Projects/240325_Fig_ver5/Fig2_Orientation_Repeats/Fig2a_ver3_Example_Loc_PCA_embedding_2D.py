'''
This script change PCA visualization into 2D space, let's see what's happenning

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

wp = r'D:\_All_Spon_Data_V1\L76_18M_220902'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
spon_series = ot.Load_Variable(wp,'Spon_Before.pkl')
# if we need raw frame dF values
# raw_orien_run = ot.Load_Variable(f'{wp}\\Orien_Frames_Raw.pkl')
# raw_spon_run = ot.Load_Variable(f'{wp}\\Spon_Before_Raw.pkl')



#%% ############################# Step0, Calculate PCA Model##################################

spon_series = np.array(spon_series)
spon_s = Spon_Shuffler(spon_series,method='phase')
# pcnum = PCNum_Determine(spon_series,sample='Frame',thres = 0.5)
pcnum = 10
g16_frames,g16_labels = ac.Combine_Frame_Labels(od = 0,color = 0,orien = 1)

# real spon models
spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=spon_series,sample='Frame',pcnum=pcnum)
model_var_ratio = np.array(spon_models.explained_variance_ratio_)
print(f'{pcnum} PCs explain Spontaneous VAR {model_var_ratio[:pcnum].sum()*100:.1f}%')


# and fit model to find spon response.
analyzer = UMAP_Analyzer(ac = ac,umap_model=spon_models,spon_frame=spon_series,od = 0,orien = 1,color = 0,isi = True)
analyzer.Train_SVM_Classifier(C=1)
stim_embed = analyzer.stim_embeddings
stim_label = analyzer.stim_label
spon_embed = analyzer.spon_embeddings
spon_label = analyzer.spon_label
# below are old shuffles
spon_s_embeddings = spon_models.transform(spon_s)
spon_label_s = SVC_Fit(analyzer.svm_classifier,spon_s_embeddings,thres_prob=0)
print(f'Spon {(spon_label>0).sum()}\nShuffle {(spon_label_s>0).sum()}')
#%% Plot PC explained variance here.
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (6,4),dpi = 144)
sns.barplot(y = model_var_ratio*100,x = np.arange(1,11),ax = ax)
ax.set_xlabel('PC',size = 12)
ax.set_ylabel('Explained Variance (%)',size = 12)
ax.set_title('Each PC explained Variance',size = 14)


#%%####################### Step1, Plot 3 embedding maps (Fig 2A)##################################
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
#%%
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
import matplotlib as mpl
import colorsys
def Plot_Colorized_Oriens_2D(axes,embeddings,labels,pcs=[4,1],color_sets = np.zeros(shape = (8,3))):
    embeddings = embeddings[:,pcs]
    rest,_ = Select_Frame(embeddings,labels,used_id=[0])
    orien,orien_id = Select_Frame(embeddings,labels,used_id=list(range(9,17)))
    orien_colors = np.zeros(shape = (len(orien_id),3),dtype='f8')
    for i,c_id in enumerate(orien_id):
        orien_colors[i,:] = color_sets[int(c_id)-9,:]
    axes.scatter(rest[:,0],rest[:,1],s = 1,c = [0.7,0.7,0.7],alpha = 0.1)
    axes.scatter(orien[:,0],orien[:,1],s = 1,c = orien_colors)
    return axes

#%%
plt.clf()
plt.cla()
plotted_pcs = [4,1]
u = spon_embed
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (4,4),dpi = 180)
# set axes
ax.set_xlabel(f'PC {plotted_pcs[0]+1}')
ax.set_ylabel(f'PC {plotted_pcs[1]+1}')
ax.grid(False)
# ax = Plot_Colorized_Oriens_2D(ax,stim_embed,stim_label,plotted_pcs,color_setb)
# ax = Plot_Colorized_Oriens_2D(ax,spon_embed,spon_label,plotted_pcs,color_setb)
ax = Plot_Colorized_Oriens_2D(ax,spon_s_embeddings,spon_label_s,plotted_pcs,color_setb)
# ax.set_title('Spontaneous in PCA Space',size = 14)
# ax.set_title('Stimulus in PCA Space',size = 14)
ax.set_title('Shuffled Spontaneous in PCA Space',size = 14)
ax.set_xlim(-20,25)
ax.set_ylim(-25,25)
fig.tight_layout()
