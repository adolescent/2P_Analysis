'''
The same as fig3, but on color.
'''

'''
Almost the same code as that in Orientation.
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
from Cell_Class.Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *
from Review_Fix_Funcs import *
from Filters import Signal_Filter_v2
import warnings

warnings.filterwarnings("ignore")

expt_folder = r'D:\_DataTemp\_Fig_Datas\_All_Spon_Data_V1\L76_18M_220902'
savepath = r'G:\我的云端硬盘\#Figs\#250211_Revision1\FigS6'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
sponrun = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
start = sponrun.index[0]
end = sponrun.index[-1]

# generate new spon series, remove LP filter.
# NOTE different shape!
spon_series = Z_refilter(ac,'1-001',start,end).T

#%%
'''
Step 0, first we need to generate the model, including svm ones
'''


spon_series = np.array(spon_series)
pcnum = 10
od_frames,od_labels = ac.Combine_Frame_Labels(od = 0,color = 1,orien = 0)
spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=spon_series,sample='Frame',pcnum=pcnum)
model_var_ratio = np.array(spon_models.explained_variance_ratio_)
print(f'{pcnum} PCs explain Spontaneous VAR {model_var_ratio[:pcnum].sum()*100:.1f}%')

# and fit model to find spon response.
analyzer = Classify_Analyzer(ac = ac,model=spon_models,spon_frame=spon_series,od = 0,orien = 0,color = 1,isi = True)
analyzer.Train_SVM_Classifier(C=1)
stim_embed = analyzer.stim_embeddings
stim_label = analyzer.stim_label
spon_embed = analyzer.spon_embeddings
spon_label = analyzer.spon_label
#%%
'''
Fig A/B/C, example of PCA embeddings, OD.
'''
def Plot_Colorized_Color(axes,embeddings,labels,pcs=[2,5,6],color_sets = np.array([[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1]])):
    embeddings = embeddings[:,pcs]
    rest,_ = Select_Frame(embeddings,labels,used_id=[0])
    color,color_ids = Select_Frame(embeddings,labels,used_id=list(range(17,23)))
    color_colors = np.zeros(shape = (len(color_ids),3),dtype='f8')
    for i,c_id in enumerate(color_ids):
        color_colors[i,:] = color_sets[int(c_id)-17,:]
    axes.scatter3D(rest[:,0],rest[:,1],rest[:,2],s = 20,lw=0,c = [0.7,0.7,0.7],alpha = 0.1)
    axes.scatter3D(color[:,0],color[:,1],color[:,2],s = 20,lw=0,c = color_colors)
    return axes
#%% Plot parts
#%% Plot parts
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
import matplotlib as mpl
import colorsys

# Color color bars.
fig = plt.figure(figsize = (3,2),dpi = 180)
color_setb = np.array([[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1]])
cax_b = fig.add_axes([-0.5, 0, 0.08, 0.9])
custom_cmap = mcolors.ListedColormap(color_setb)
bounds = np.arange(0,7,1)
norm = mpl.colors.BoundaryNorm(bounds, custom_cmap.N)
c_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=custom_cmap),cax=cax_b, label='Best Color')
c_bar.set_ticks(np.arange(0,6,1)+0.5)
c_bar.set_ticklabels(['Red','Yellow','Green','Cyan','Blue','Purple'])
c_bar.ax.tick_params(size=0)
c_bar.ax.tick_params(labelsize=8)
c_bar.set_label(label='Best Color',size=10)
#%% Plot graphs here.
plotted_pcs = [2,3,4]
orien_elev = 15
orien_azim = 40
zoom = 1
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (6,6),dpi = 300,subplot_kw=dict(projection='3d'))
# Grid Preparing
# ax.set_xlabel(f'PC {plotted_pcs[0]+1}')
# ax.set_ylabel(f'PC {plotted_pcs[1]+1}')
# ax.set_zlabel(f'PC {plotted_pcs[2]+1}')
ax.grid(False)
ax.view_init(elev=orien_elev, azim=orien_azim)
ax.set_box_aspect(aspect=None, zoom=1) # shrink graphs
ax.axes.set_xlim3d(left=-15, right=20)
ax.axes.set_ylim3d(bottom=-20, top=20)
ax.axes.set_zlim3d(bottom=-20, top=20)
# ax[i].set_position([ax[i].get_position().x0-0.12, ax[i].get_position().y0, ax[i].get_position().width*zoom, ax[i].get_position().height*zoom])
# set z label location
tmp_planes = ax.zaxis._PLANES 
ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                        tmp_planes[0], tmp_planes[1], 
                        tmp_planes[4], tmp_planes[5])
    

ax = Plot_Colorized_Color(ax,stim_embed,stim_label,plotted_pcs,color_setb)
# ax = Plot_Colorized_Color(ax,spon_embed,spon_label,plotted_pcs,color_setb)
# ax = Plot_Colorized_Color(ax,spon_embed,np.zeros(len(spon_label)),plotted_pcs,color_setb)
# set title
# ax.set_title('OD Stimulus in PCA Space',size = 10)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
fig.tight_layout()
fig.savefig(ot.join(savepath,'S3B_Stim_Color.png'),bbox_inches='tight')
# fig.savefig(ot.join(savepath,'S3C_Spon_Classified.png'),bbox_inches='tight')
# fig.savefig(ot.join(savepath,'S3A_Spon_Raw.png'),bbox_inches='tight')

#%%
'''
Fig I/J, we get recovered graph and compare it with real stim graph
'''
value_max = 3
value_min = -1

analyzer.Get_Stim_Spon_Compare(od = False,color = True,orien = False)
stim_graphs = analyzer.stim_recover
spon_graphs = analyzer.spon_recover
graph_lists = ['Red','Green','Blue']
analyzer.Similarity_Compare_Average(od = False,color = True,orien = False)
all_corr = analyzer.Avr_Similarity

plt.clf()
plt.cla()
# cbar_ax = fig.add_axes([.92, .45, .01, .2])
font_size = 16
fig,axes = plt.subplots(nrows=1, ncols=3,figsize = (10.5,4),dpi = 300)
for i,c_map in enumerate(graph_lists):
    sns.heatmap(spon_graphs[c_map][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[i],vmax = value_max,vmin = value_min,cbar=False,square=True)

fig.tight_layout()

# fig.savefig(ot.join(savepath,'S3E_Color_Stim_Graph.png'),bbox_inches='tight')
fig.savefig(ot.join(savepath,'S3D_Color_Spon_Graph.png'),bbox_inches='tight')

#%% print R values here.
for i,c_graph in enumerate(graph_lists):
    print(f'Graph {c_graph}, R = {all_corr.iloc[i*2,0]:.3f}')