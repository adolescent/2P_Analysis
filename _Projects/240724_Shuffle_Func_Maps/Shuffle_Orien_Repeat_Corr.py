'''
This script will do pca-svm on shuffled G16 graph, we will find no good repeat on shuffled graph.
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
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import cross_val_score
from sklearn import svm
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
import random
from Classifier_Analyzer import *

wp = r'D:\_Path_For_Figs\240724_Figs_Graph_Shuffle'
orien_shuffled = ot.Load_Variable(wp,'Shuffle_FuncMaps.pkl')
ac = ot.Load_Variable(r'D:\_All_Spon_Data_V1\L76_18M_220902\Cell_Class.pkl')
ac_locs = ac.Cell_Locs
spon_series = np.array(ot.Load_Variable(r'D:\_All_Spon_Data_V1\L76_18M_220902','Spon_Before.pkl'))

#%% 
'''
Part 1, plot example of original cell and blured-shuffle sequence.
'''
# use orien 0 and shuffle time 1 as example.
def Graph_Filler(ac_locs,cell_resp,extend = 40,clip = 5):
    response_frame = np.zeros(shape = (512,512),dtype = 'f8')
    clipped_cell_resp = np.clip(cell_resp,cell_resp.mean()-cell_resp.std()*clip,cell_resp.mean()+cell_resp.std()*clip)
    for i,c_response in enumerate(clipped_cell_resp):
        y_cord,x_cord = ac_locs[i+1]
        # y_min = max(y_cord-extend,0)
        # y_max = min(y_cord+extend,511)
        # x_min = max(x_cord-extend,0)
        # x_max = min(x_cord+extend,511)
        x = np.arange(512)
        y = np.arange(512)
        X, Y = np.meshgrid(x, y)
        gaussian = np.exp(-((X-x_cord)**2+(Y-y_cord)**2)/(2*extend**2))
        # response_frame[y_min:y_max,x_min:x_max] = c_response
        gaussian[gaussian<0.2] = 0
        response_frame += c_response*gaussian
    return response_frame

o0_resp = ac.Orien_t_graphs['Orien0-0'].loc['CohenD']
o22_resp = ac.Orien_t_graphs['Orien22.5-0'].loc['CohenD']
o45_resp = ac.Orien_t_graphs['Orien45-0'].loc['CohenD']
o67_resp = ac.Orien_t_graphs['Orien67.5-0'].loc['CohenD']
o90_resp = ac.Orien_t_graphs['Orien90-0'].loc['CohenD']
o112_resp = ac.Orien_t_graphs['Orien112.5-0'].loc['CohenD']
o135_resp = ac.Orien_t_graphs['Orien135-0'].loc['CohenD']
o157_resp = ac.Orien_t_graphs['Orien157.5-0'].loc['CohenD']

o0_frame = ac.Generate_Weighted_Cell(o0_resp)
o0_blured = Graph_Filler(ac_locs,o0_resp)
exp_o0_shuffle = ac.Generate_Weighted_Cell(orien_shuffled[1][0])
exp_o0_shuffle_blured = Graph_Filler(ac_locs,orien_shuffled[1][0])

#%% Plot example parts
plotable = exp_o0_shuffle
vmax = 3
vmin = -2

plt.clf()
plt.cla()

fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (5,5),dpi = 300)
sns.heatmap(plotable,ax = ax,center = 0,cmap='gist_gray',square=True,xticklabels=False,yticklabels=False,cbar=False)


#%% Plot color bar seperetly.
plt.clf()
plt.cla()
data = [[vmin, vmax], [vmin, vmax]]
# Create a heatmap
fig, ax = plt.subplots(figsize = (2,1),dpi = 600)
# fig2, ax2 = plt.subplots()
g = sns.heatmap(data, center=0,ax = ax,vmax = vmax,vmin = vmin,cbar_kws={"aspect": 5,"shrink": 1,"orientation": "vertical"},cmap='gist_gray')
# Hide the heatmap itself by setting the visibility of its axes
ax.set_visible(False)
g.collections[0].colorbar.set_ticks([vmin,0,vmax])
g.collections[0].colorbar.set_ticklabels([vmin,0,vmax])
g.collections[0].colorbar.ax.tick_params(labelsize=14)
plt.show()
#%%
'''
Part 2, we show example of shuffled stim map's recover
'''
exp_all_maps = orien_shuffled[1]
all_cell_std = spon_series.std(0)
g16_frames,g16_labels = ac.Combine_Frame_Labels(od = 0,color = 0,orien = 1)
g16_frames = np.array(g16_frames)

for i,c_id in tqdm(enumerate(g16_labels)):
    if c_id != 0: # if not isi, we will replace real response with 
        base = copy.deepcopy(exp_all_maps[c_id-9])
        vars = np.random.normal(loc=0, scale=all_cell_std, size=len(base))
        c_shuffle = base + vars
        g16_frames[i,:] = c_shuffle

# and we train svm use shuffled info.
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
#%% Plot p1,plot 3D embedding first, use original function.
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
import matplotlib as mpl
import colorsys

color_setb = np.zeros(shape = (8,3))
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
    axes.scatter3D(rest[:,0],rest[:,1],rest[:,2],s = 10,c = [0.7,0.7,0.7],alpha = 0.1,lw=0)
    axes.scatter3D(orien[:,0],orien[:,1],orien[:,2],s = 10,c=orien_colors,lw=0,alpha = 0.5)
    return axes

plt.clf()
plt.cla()
plotted_pcs = [4,1,2]
u = spon_embed
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (6,6),dpi = 300,subplot_kw=dict(projection='3d'))
orien_elev = 25
orien_azim = -100
# set axes
# ax.set_xlabel(f'PC {plotted_pcs[0]+1}')
# ax.set_ylabel(f'PC {plotted_pcs[1]+1}')
# ax.set_zlabel(f'PC {plotted_pcs[2]+1}')
ax.grid(False)
ax.view_init(elev=orien_elev, azim=orien_azim)
ax.set_box_aspect(aspect=None, zoom=1) # shrink graphs
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

# ax.set_title('Classified Spontaneous in PCA Space',size = 10)
# ax.set_title('Orientation Stimulus in PCA Space',size = 10)
# ax.set_title('Shuffled Spontaneous in PCA Space',size = 10)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
fig.tight_layout()
#%% Plot P2, average recovered spon graph and compare it with shuffled stim graph.
used_id = [9,11,13,15]
stim_rec = []
spon_rec = []
rs = []
for i,c_id in enumerate(used_id):
    c_spon_rec,_ = Select_Frame(frame = spon_series,label = spon_label,used_id=[c_id])
    c_stim_rec,_ = Select_Frame(frame = g16_frames,label = stim_label,used_id=[c_id])
    stim_rec.append(c_stim_rec.mean(0))
    spon_rec.append(c_spon_rec.mean(0))
    r,p = stats.pearsonr(c_stim_rec.mean(0),c_spon_rec.mean(0))
    rs.append(r)
#%% Plot example of shuffled graph and recovered shuffled graph.
value_max = 3
value_min = -1

plt.clf()
plt.cla()
# cbar_ax = fig.add_axes([.92, .45, .01, .2])
font_size = 14
fig,axes = plt.subplots(nrows=1, ncols=4,figsize = (14,4),dpi = 180)
for i,c_map in enumerate(spon_rec):
    plotable = ac.Generate_Weighted_Cell(c_map)
    sns.heatmap(plotable,center = 0,xticklabels=False,yticklabels=False,ax = axes[i],vmax = value_max,vmin = value_min,cbar=False,square=True)

fig.tight_layout()
# axes[0].set_ylabel('Spontaneous',rotation=90,size = font_size)

# fig.tight_layout()
plt.show()
#%% Plot bars
plt.clf()
plt.cla()
data = [[value_min, value_max], [value_min, value_max]]
# Create a heatmap
fig, ax = plt.subplots(figsize = (2,1),dpi = 300)
# fig2, ax2 = plt.subplots()
g = sns.heatmap(data, center=0,ax = ax,vmax = value_max,vmin = value_min,cbar_kws={"aspect": 5,"shrink": 1,"orientation": "horizontal"})
# Hide the heatmap itself by setting the visibility of its axes
ax.set_visible(False)
g.collections[0].colorbar.set_ticks([value_min,0,value_max])
g.collections[0].colorbar.set_ticklabels([value_min,0,value_max])
g.collections[0].colorbar.ax.tick_params(labelsize=12)
plt.show()

#%%
'''
Part 3, we stat the result of all 100 shuffle's correlation index.
'''
all_r_frame = pd.DataFrame(columns = ['Pearson R','Network','Data Type'])
for n in tqdm(range(100)):
    c_all_maps = orien_shuffled[n]
    all_cell_std = spon_series.std(0)
    g16_frames,g16_labels = ac.Combine_Frame_Labels(od = 0,color = 0,orien = 1)
    g16_frames = np.array(g16_frames)
    for i,c_id in tqdm(enumerate(g16_labels)):
        if c_id != 0: # if not isi, we will replace real response with 
            base = copy.deepcopy(exp_all_maps[c_id-9])
            vars = np.random.normal(loc=0, scale=all_cell_std, size=len(base))
            c_shuffle = base + vars
            g16_frames[i,:] = c_shuffle
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
    # stat all rs.
    used_id = [9,11,13,15]

    for i,c_id in enumerate(used_id):
        c_orien = (c_id-9)*45//2
        c_spon_rec,_ = Select_Frame(frame = spon_series,label = spon_label,used_id=[c_id])
        c_stim_rec,_ = Select_Frame(frame = g16_frames,label = stim_label,used_id=[c_id])
        r,p = stats.pearsonr(c_stim_rec.mean(0),c_spon_rec.mean(0))
        all_r_frame.loc[len(all_r_frame)] = [r,c_orien,'Shuffled']

#%% Plot stats of all rs.
all_r_frame.loc[len(all_r_frame)] = [0.68,0,'Real Spon']
all_r_frame.loc[len(all_r_frame)] = [0.68,45,'Real Spon']
all_r_frame.loc[len(all_r_frame)] = [0.80,90,'Real Spon']
all_r_frame.loc[len(all_r_frame)] = [0.80,135,'Real Spon']

plt.clf()
plt.cla()
fontsize = 14
plotable = all_r_frame
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (3,5),dpi = 300)
sns.barplot(data = plotable,ax = ax,width=0.5,hue='Data Type',x = 'Network',y = 'Pearson R',legend=False,errwidth=1)



ax.set_ylim(0,1)
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=fontsize)
ax.set_xticks([])
