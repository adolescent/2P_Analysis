'''
This graph shows the umap embedded stim and spon frames.

Here we use class method to avoid dump repeatance.

Example location: L76_18M_220902
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


work_path = r'D:\_Path_For_Figs\_2312_ver2\Fig2'
expt_folder = r'D:\_All_Spon_Datas_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
ac.wp = expt_folder
spon_series = ot.Load_Variable(expt_folder,'Spon_Before.pkl')

# load reducer, if not exist, generate a new one.
reducer = ot.Load_Variable_v2(expt_folder,'All_Stim_UMAP_3D_20comp.pkl')
if reducer == False:
    raise ValueError('No reducer file, you need to generate it first.')
#%%#################### STEP1, GET EMBEDDING SERIES ###############################
# do svm prediction and get stim-spon embeddings.
analyzer = UMAP_Analyzer(ac = ac,umap_model=reducer,spon_frame=spon_series,od = True,orien = True,color = True,isi = True)
analyzer.Train_SVM_Classifier()




#%%#################### STEP2, PLOT ALL EMBEDDING GRAPHS ###################################

# This time we plot only od and orientation graphs, with other situation in gray alpha 0.5.
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
import matplotlib as mpl
import colorsys
# define several functions here.

plt.clf()
plt.cla()
fig,axes = plt.subplots(nrows=3, ncols=2,figsize = (9,14),dpi = 180,subplot_kw=dict(projection='3d'))
# Line 0: OD graphs, only OD colorized
# Line 1: Orien graphs, only Orientation colorized
# Line 2: Color graphs, only hue colorized
for i in range(3):
    for j in range(2):
        axes[i,j].grid(False)
        axes[i,j].grid(False)
#### plot OD graphs with only od color bar.
od_elev = 30
od_azim = 240
## part stim
axes[0,0].view_init(elev=od_elev, azim=od_azim)
axes[0,1].view_init(elev=od_elev, azim=od_azim)
rest_stim,_ = Select_Frame(stim_embed,stim_label,used_id=list(range(9,23))+[0])
axes[0,0].scatter3D(rest_stim[:,0],rest_stim[:,1],rest_stim[:,2],s = 1,c = [0.7,0.7,0.7],alpha = 0.1)
LE_stim,_ = Select_Frame(stim_embed,stim_label,used_id=[1,3,5,7])
axes[0,0].scatter3D(LE_stim[:,0],LE_stim[:,1],LE_stim[:,2],s = 1,c = [1,0,0])
RE_stim,_ = Select_Frame(stim_embed,stim_label,used_id=[2,4,6,8])
axes[0,0].scatter3D(RE_stim[:,0],RE_stim[:,1],RE_stim[:,2],s = 1,c = [0,1,0])
## part spon
rest_spon,_ = Select_Frame(spon_embed,spon_label,used_id=list(range(9,23))+[0])
axes[0,1].scatter3D(rest_spon[:,0],rest_spon[:,1],rest_spon[:,2],s = 1,c = [0.7,0.7,0.7],alpha = 0.1)
LE_spon,_ = Select_Frame(spon_embed,spon_label,used_id=[1,3,5,7])
axes[0,1].scatter3D(LE_spon[:,0],LE_spon[:,1],LE_spon[:,2],s = 1,c = [1,0,0])
RE_spon,_ = Select_Frame(spon_embed,spon_label,used_id=[2,4,6,8])
axes[0,1].scatter3D(RE_spon[:,0],RE_spon[:,1],RE_spon[:,2],s = 1,c = [0,1,0])
## and color bar here.
cax_a = fig.add_axes([0.965, 0.72, 0.02, 0.15])
color_seta = np.array([[1,0,0],[0,1,0]])
custom_cmap = mcolors.ListedColormap(color_seta)
bounds = np.arange(0,3,1)
norm = mpl.colors.BoundaryNorm(bounds, custom_cmap.N)
c_bar_a = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=custom_cmap),cax=cax_a, label='Eye')
c_bar_a.set_ticks(np.arange(0,2,1)+0.5)
c_bar_a.set_ticklabels(['LE','RE'])
c_bar_a.ax.tick_params(size=0)
## limit and title.
for i in range(2):
    for j in range(3):
        axes[j,i].set_xlabel('UMAP 1')
        axes[j,i].set_ylabel('UMAP 2')
        axes[j,i].set_zlabel('UMAP 3')
        axes[j,i].axes.set_xlim3d(left=1, right=14) 
        axes[j,i].axes.set_ylim3d(bottom=2, top=14) 
        axes[j,i].axes.set_zlim3d(bottom=5, top=15) 
axes[0,0].set_title('Stimulus Embedding in UMAP Space',size = 14)
axes[0,1].set_title('Spontaneous Embedding in UMAP Space',size = 14)

##### Orientation on line 2.
orien_elev = 55
orien_azim = 30
axes[1,0].view_init(elev=orien_elev, azim=orien_azim)
axes[1,1].view_init(elev=orien_elev, azim=orien_azim)
## generate color set and color bar.
color_setb = np.zeros(shape = (8,3))
for i,c_orien in enumerate(np.arange(0,180,22.5)):
    c_hue = c_orien/180
    c_lightness = 0.5
    c_saturation = 1
    color_setb[i,:] = colorsys.hls_to_rgb(c_hue, c_lightness, c_saturation)
# below is color bar.
cax_b = fig.add_axes([0.965, 0.4, 0.02, 0.15])
custom_cmap = mcolors.ListedColormap(color_setb)
bounds = np.arange(0,202.5,22.5)
norm = mpl.colors.BoundaryNorm(bounds, custom_cmap.N)
c_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=custom_cmap),cax=cax_b, label='Orientation')
c_bar.set_ticks(np.arange(0,180,22.5)+11.25)
c_bar.set_ticklabels(np.arange(0,180,22.5))
c_bar.ax.tick_params(size=0)
## part stim
rest_stim,_ = Select_Frame(stim_embed,stim_label,used_id=list(range(17,23))+list(range(0,9)))
orien_stim,orien_stim_id = Select_Frame(stim_embed,stim_label,used_id=list(range(9,17)))
# get color system of stim.
orien_color_stim = np.zeros(shape = (len(orien_stim_id),3),dtype='f8')
for i,c_id in enumerate(orien_stim_id):
    orien_color_stim[i,:] = color_setb[int(c_id)-9,:]
axes[1,0].scatter3D(rest_stim[:,0],rest_stim[:,1],rest_stim[:,2],s = 1,c = [0.7,0.7,0.7],alpha = 0.1)
axes[1,0].scatter3D(orien_stim[:,0],orien_stim[:,1],orien_stim[:,2],s = 1,c = orien_color_stim)
## part spon
rest_spon,_ = Select_Frame(spon_embed,spon_label,used_id=list(range(17,23))+list(range(0,9)))
orien_spon,orien_spon_id = Select_Frame(spon_embed,spon_label,used_id=list(range(9,17)))
# get color system of stim.
orien_color_spon = np.zeros(shape = (len(orien_spon_id),3),dtype='f8')
for i,c_id in enumerate(orien_spon_id):
    orien_color_spon[i,:] = color_setb[int(c_id)-9,:]
axes[1,1].scatter3D(rest_spon[:,0],rest_spon[:,1],rest_spon[:,2],s = 1,c = [0.7,0.7,0.7],alpha = 0.1)
axes[1,1].scatter3D(orien_spon[:,0],orien_spon[:,1],orien_spon[:,2],s = 1,c = orien_color_spon)


##### Color on line 3.
color_elev = 25
color_azim = 220
axes[2,0].view_init(elev=color_elev, azim=color_azim)
axes[2,1].view_init(elev=color_elev, azim=color_azim)
## generate color set and color bar.
color_setc = np.array([[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1]])
cax_c = fig.add_axes([0.965, 0.1, 0.02, 0.15])
custom_cmap = mcolors.ListedColormap(color_setc)
bounds = np.arange(0,7,1)
norm = mpl.colors.BoundaryNorm(bounds, custom_cmap.N)
c_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=custom_cmap),cax=cax_c, label='Color')
c_bar.set_ticks(np.arange(0,6,1)+0.5)
c_bar.set_ticklabels(['Red','Yellow','Green','Cyan','Blue','Purple'])
c_bar.ax.tick_params(size=0)

## part stim
rest_color_stim,_ = Select_Frame(stim_embed,stim_label,used_id=list(range(0,17)))
color_stim,color_stim_id = Select_Frame(stim_embed,stim_label,used_id=list(range(17,23)))
hue_color_stim = np.zeros(shape = (len(color_stim_id),3),dtype='f8')
for i,c_id in enumerate(color_stim_id):
    hue_color_stim[i,:] = color_setc[int(c_id)-17,:]
axes[2,0].scatter3D(rest_color_stim[:,0],rest_color_stim[:,1],rest_color_stim[:,2],s = 1,c = [0.7,0.7,0.7],alpha = 0.1)
axes[2,0].scatter3D(color_stim[:,0],color_stim[:,1],color_stim[:,2],s = 1,c = hue_color_stim)
## part spon
rest_color_spon,_ = Select_Frame(spon_embed,spon_label,used_id=list(range(0,17)))
color_spon,color_spon_id = Select_Frame(spon_embed,spon_label,used_id=list(range(17,23)))
hue_color_spon = np.zeros(shape = (len(color_spon_id),3),dtype='f8')
for i,c_id in enumerate(color_spon_id):
    hue_color_spon[i,:] = color_setc[int(c_id)-17,:]
axes[2,1].scatter3D(rest_color_spon[:,0],rest_color_spon[:,1],rest_color_spon[:,2],s = 1,c = [0.7,0.7,0.7],alpha = 0.1)
axes[2,1].scatter3D(color_spon[:,0],color_spon[:,1],color_spon[:,2],s = 1,c = hue_color_spon)


#### global adjust.
for i in range(3):
    for j in range(2):
        axes[i,j].set_box_aspect(aspect=None, zoom=0.86)

plt.subplots_adjust(left=0.1, right=0.95,top = 0.93,bottom=0.1)
fig.suptitle('Stimulus & Spon Embeddings',size = 24,x = 0.5,y = 1)
plt.tight_layout()
# plt.show()