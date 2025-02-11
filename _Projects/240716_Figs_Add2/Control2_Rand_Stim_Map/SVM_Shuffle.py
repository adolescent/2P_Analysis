'''
This script will do PCA-SVM on shuffled data. 

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
save_path = r'D:\_GoogleDrive_Files\#Figs\240717_Controls\Dim_Control'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
spon_series = ot.Load_Variable(wp,'Spon_Before.pkl')
shuffle_orien_info = ot.Load_Variable(save_path,'Cell_Shuffle.pkl')
#%%
'''
Part 0 we will show shuffled cell response, make sure it's okay to plot.
'''
shuffled_cells = shuffle_orien_info['Shuffle_Cell']

plt.clf()
plt.cla()
value_max = 3
value_min = -2
font_size = 13
fig,axes = plt.subplots(nrows=2, ncols=4,figsize = (12,6),dpi = 180)
cbar_ax = fig.add_axes([1, .45, .01, .2])
for i in range(8):
    c_map = ac.Generate_Weighted_Cell(shuffled_cells[i])
    # sns.heatmap(c_pc_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[i//5,i%5],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True,cmap = cmaps.pinkgreen_light)
    sns.heatmap(c_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[i//4,i%4],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True,cmap = 'gist_gray')
    # axes[i//4,i%4].set_title(f'PC {i+1}',size = font_size)

fig.tight_layout()


#%%
'''
Part 1, we will get shuffled spon data set. Make fake stims with rands.
'''
all_cell_std = spon_series.std(0)
g16_frames,g16_labels = ac.Combine_Frame_Labels(od = 0,color = 0,orien = 1)
g16_frames = np.array(g16_frames)
used_ids = []
for i,c_id in tqdm(enumerate(g16_labels)):
    if c_id != 0: # if not isi, we will shuffle
        base = copy.deepcopy(shuffle_orien_info['Shuffle_Cell'][c_id-9])
        vars = np.random.normal(loc=0, scale=all_cell_std, size=len(base))
        c_shuffle = base + vars
        g16_frames[i,:] = c_shuffle
    if (c_id == 10) or (c_id == 13) or (c_id == 15) or (c_id == 16) or (c_id == 0):
        used_ids.append(i)

used_g16_frames = g16_frames[used_ids,:]
used_g16_labels = g16_labels[used_ids]

#%% and use this to train an PCA-SVM, we will see embeddings here.
pcnum = 10
spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=spon_series,sample='Frame',pcnum=pcnum)
analyzer = Classify_Analyzer(ac = ac,model=spon_models,spon_frame=spon_series,od = 0,orien = 1,color = 0,isi = True)
analyzer.stim_frame = g16_frames
analyzer.stim_label = g16_labels
analyzer.stim_embeddings = analyzer.model.transform(analyzer.stim_frame)
analyzer.Train_SVM_Classifier(C=1)

stim_embed = analyzer.stim_embeddings
stim_label = analyzer.stim_label
spon_embed = analyzer.spon_embeddings
spon_label = analyzer.spon_label
#%% 
'''
3D Plot of SVM shuffled graph.
'''
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
import matplotlib as mpl
import colorsys

color_setb = np.zeros(shape = (8,3))
fig = plt.figure(figsize = (2,4),dpi = 180)
for i,c_orien in enumerate(np.arange(0,180,22.5)):
    c_hue = c_orien/180
    c_lightness = 0.5
    c_saturation = 1
    color_setb[i,:] = colorsys.hls_to_rgb(c_hue, c_lightness, c_saturation)


def Plot_Colorized_Oriens(axes,embeddings,labels,pcs=[4,1,2],color_sets = np.zeros(shape = (8,3))):
    embeddings = embeddings[:,pcs]
    rest,_ = Select_Frame(embeddings,labels,used_id=[0])
    orien,orien_id = Select_Frame(embeddings,labels,used_id=list(range(9,17)))
    orien_colors = np.zeros(shape = (len(orien_id),3),dtype='f8')
    for i,c_id in enumerate(orien_id):
        orien_colors[i,:] = color_sets[int(c_id)-9,:]
    axes.scatter3D(rest[:,0],rest[:,1],rest[:,2],s = 1,c = [0.7,0.7,0.7],alpha = 0.1)
    axes.scatter3D(orien[:,0],orien[:,1],orien[:,2],s = 1,c = orien_colors)
    return axes

plt.clf()
plt.cla()
plotted_pcs = [4,1,2]
u = spon_embed
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (8,4),dpi = 180,subplot_kw=dict(projection='3d'))
orien_elev = 25
orien_azim = -60
# set axes
ax.set_xlabel(f'PC {plotted_pcs[0]+1}')
ax.set_ylabel(f'PC {plotted_pcs[1]+1}')
ax.set_zlabel(f'PC {plotted_pcs[2]+1}')
ax.grid(False)
ax.view_init(elev=orien_elev, azim=orien_azim)
ax.set_box_aspect(aspect=None, zoom=0.85) # shrink graphs
# ax.axes.set_xlim3d(left=-20, right=30)
# ax.axes.set_ylim3d(bottom=-30, top=40)
# ax.axes.set_zlim3d(bottom=20, top=-20)
# tmp_planes = ax.zaxis._PLANES 
# ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
#                         tmp_planes[0], tmp_planes[1], 
#                         tmp_planes[4], tmp_planes[5])
# ax = Plot_Colorized_Oriens(ax,spon_embed,np.zeros(len(spon_embed)),plotted_pcs,color_setb)
# ax = Plot_Colorized_Oriens(ax,stim_embed,stim_label,plotted_pcs,color_setb)
ax = Plot_Colorized_Oriens(ax,spon_embed,spon_label,plotted_pcs,color_setb)
# ax = Plot_Colorized_Oriens(ax,spon_s_embeddings,spon_label_s,plotted_pcs,color_setb)
# ax.set_title('Classified Spontaneous in PCA Space',size = 10)
# ax.set_title('Orientation Stimulus in PCA Space',size = 10)
# ax.set_title('Shuffled Spontaneous in PCA Space',size = 10)

fig.tight_layout()
#%%
Orien22_corr,_,Orien22_corr_num = analyzer.Average_Corr_Core([9])
Orien67_corr,_,Orien67_corr_num = analyzer.Average_Corr_Core([11])
Orien112_corr,_,Orien112_corr_num = analyzer.Average_Corr_Core([13])
Orien157_corr,_,Orien157_corr_num = analyzer.Average_Corr_Core([15])

o22_recover,_ = Select_Frame(frame = analyzer.spon_frame,label = analyzer.spon_label,used_id=[9])
o67_recover,_ = Select_Frame(frame = analyzer.spon_frame,label = analyzer.spon_label,used_id=[11])
o112_recover,_ = Select_Frame(frame = analyzer.spon_frame,label = analyzer.spon_label,used_id=[13])
o157_recover,_ = Select_Frame(frame = analyzer.spon_frame,label = analyzer.spon_label,used_id=[15])
all_corr = [Orien22_corr,Orien67_corr,Orien112_corr,Orien157_corr]

o22_graph = ac.Generate_Weighted_Cell(o22_recover.mean(0))
o67_graph = ac.Generate_Weighted_Cell(o67_recover.mean(0))
o112_graph = ac.Generate_Weighted_Cell(o112_recover.mean(0))
o157_graph = ac.Generate_Weighted_Cell(o157_recover.mean(0))

#%% Plot compare graph here.
plt.clf()
plt.cla()
value_max = 3
value_min = -1
font_size = 16
fig,axes = plt.subplots(nrows=2, ncols=4,figsize = (14,7),dpi = 180)
cbar_ax = fig.add_axes([.92, .45, .01, .2])

for i in range(4):
    sns.heatmap([o22_graph,o67_graph,o112_graph,o157_graph][i],center = 0,xticklabels=False,yticklabels=False,ax = axes[1,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)

    c_stim = shuffle_orien_info['Shuffle_Cell'][2*i+1]
    sns.heatmap(ac.Generate_Weighted_Cell(c_stim),center = 0,xticklabels=False,yticklabels=False,ax = axes[0,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True,cbar_kws={'label': 'Z Scored Activity'})
    # axes[0,i].set_title(c_map,size = font_size)

axes[1,0].set_ylabel('Spon',rotation=90,size = font_size)
axes[0,0].set_ylabel('Stim',rotation=90,size = font_size)

# fig.tight_layout()
plt.show()

#%%
'''
Method 2, we compare shuffled stim map with spon directly. This might be significantly lower than best oriens disp.
'''
real_stimmaps = shuffle_orien_info['Cell_Resp']
shuffle_stimmaps = shuffle_orien_info['Shuffle_Cell']
spon_series = np.array(spon_series)
all_corr_frame = pd.DataFrame(0.0,columns = ['Real_Corr','Shuffle_Corr'],index = range(len(spon_series)))

for i in tqdm(range(len(spon_series))):
    c_map = spon_series[i,:]
    c_real_corrs=np.zeros(len(real_stimmaps))
    c_shuffle_corrs=np.zeros(len(shuffle_stimmaps))
    for j in range(len(c_real_corrs)):
        c_real_corrs[j],_ = stats.pearsonr(c_map,real_stimmaps[j])
        c_shuffle_corrs[j],_ = stats.pearsonr(c_map,shuffle_stimmaps[j])
    all_corr_frame.loc[i,:] = [abs(c_real_corrs).max(),abs(c_shuffle_corrs).max()]
#%% Plot parts
    
plotable = copy.deepcopy(all_corr_frame)

plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (5,5),dpi = 180)

ax.plot([0,0.7],[0,0.7],linestyle = '--',color = 'gray',alpha = 0.5)
sns.scatterplot(data = plotable,x = 'Real_Corr',y = 'Shuffle_Corr',ax = ax,s = 2,lw=0)
ax.set_ylim(0,0.7)
ax.set_xlim(0,0.7)
