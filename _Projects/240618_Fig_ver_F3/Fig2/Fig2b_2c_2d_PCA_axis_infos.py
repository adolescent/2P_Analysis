'''
This script will show example of PCA axes, and we will get all network similarity .

New Fig 2 will NOT use SVM, so the code here is a little different.
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
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
spon_series = ot.Load_Variable(wp,'Spon_Before.pkl')

savepath = r'D:\_GoogleDrive_Files\#Figs\240627_Figs_FF1\Fig2'

#%%
'''
Fig2B, we do pca on given spon matrix, and we show embedding of pc1-3.
2B-2 will show the explained var of given location.

We will also Generate shuffled ones, if we plot using shuffle, this will be S2b.
'''
spon_series = np.array(spon_series)
spon_s = Spon_Shuffler(spon_series,method='phase')
pcnum = 10

# real spon models
spon_pcs,spon_coords,spon_model = Z_PCA(Z_frame=spon_series,sample='Frame',pcnum=pcnum)
model_var_ratio = np.array(spon_model.explained_variance_ratio_)
print(f'{pcnum} PCs explain Spontaneous VAR {model_var_ratio[:pcnum].sum()*100:.1f}%')

spon_pcs_s,spon_coords_s,spon_model_s = Z_PCA(Z_frame=spon_s,sample='Frame',pcnum=pcnum)
model_var_ratio_s = np.array(spon_model_s.explained_variance_ratio_)
print(f'{pcnum} PCs explain Spontaneous VAR {model_var_ratio_s[:pcnum].sum()*100:.1f}%')

#%% Plot raw embeddings of previous 3 PCs.
u = spon_coords_s[:,:3]

plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (6,6),dpi = 300,subplot_kw=dict(projection='3d'))
orien_elev = 25
orien_azim = 50
# set axes
# ax.set_xlabel(f'PC 1')
# ax.set_ylabel(f'PC 2')
# ax.set_zlabel(f'PC 3')
ax.grid(False)
ax.view_init(elev=orien_elev, azim=orien_azim)
ax.set_box_aspect(aspect=None, zoom=1) # shrink graphs
ax.axes.set_xlim3d(left=-40, right=40)
ax.axes.set_ylim3d(bottom=-30, top=30)
ax.axes.set_zlim3d(bottom=30, top=-30)
ax.scatter3D(u[:,0],u[:,1],u[:,2],s = 20,c = [0.7,0.7,0.7],alpha = 1,lw = 0)
# ax.set_title('Shuffled in PCA Space',size = 10)
# ax.set_xticks(np.arange(-40,50,20))
# ax.set_yticks(np.arange(-30,50,20))
# ax.set_zticks(np.arange(-30,40,20))
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
fig.tight_layout()
# fig.savefig(ot.join(savepath,'Fig2B_Spon_Embedding.svg'))

#%% Plot PCA explained VARs.
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (6,4),dpi = 180)
sns.barplot(y = model_var_ratio_s*100,x = np.arange(1,11),ax = ax)
ax.set_xlabel('PC',size = 12)
ax.set_ylabel('Explained Variance (%)',size = 12)
# ax.set_title('Each PC explained Variance',size = 14)
ax.set_ylim(0,37)

# fig.savefig(ot.join(savepath,'test.png'))



#%%
'''
Fig 2D & S2D, we will generate PCA main axis, and compare it with stimulus maps. Different from origional version,we plot grayscale plots here.
'''
# only PC graph, it's very easy.
# import cmasher as cmr
# import colormaps as cmaps

#%% Plot bars seperetly.
plt.clf()
plt.cla()
vmax = 4
vmin = -4
data = [[vmin, vmax], [vmin, vmax]]
# Create a heatmap
fig, ax = plt.subplots(figsize = (2,1),dpi = 300)
# fig2, ax2 = plt.subplots()
# g = sns.heatmap(data, center=0,ax = ax,vmax = vmax,vmin = vmin,cbar_kws={"aspect": 5,"shrink": 1,"orientation": "horizontal"},cmap = 'gist_gray')
g = sns.heatmap(data, center=0,ax = ax,vmax = vmax,vmin = vmin,cbar_kws={"aspect": 5,"shrink": 1,"orientation": "horizontal"})
# Hide the heatmap itself by setting the visibility of its axes
ax.set_visible(False)
g.collections[0].colorbar.set_ticks([vmin,0,vmax])
g.collections[0].colorbar.set_ticklabels([vmin,0,vmax])
g.collections[0].colorbar.ax.tick_params(labelsize=8)
plt.show()

#%% Plot PC comps.
plt.clf()
plt.cla()
value_max = 0.1
value_min = -0.1
font_size = 13
fig,axes = plt.subplots(nrows=2, ncols=5,figsize = (12,6),dpi = 300)
# cbar_ax = fig.add_axes([1, .45, .01, .2])
for i in tqdm(range(10)):
    c_pc = spon_pcs_s[i,:]
    c_pc_graph = ac.Generate_Weighted_Cell(c_pc)
    # sns.heatmap(c_pc_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[i//5,i%5],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True,cmap = cmaps.pinkgreen_light)
    sns.heatmap(c_pc_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[i//5,i%5],vmax = value_max,vmin = value_min,cbar= False,square=True,cmap = 'gist_gray')
    # axes[i//5,i%5].set_title(f'PC {i+1}',size = font_size)

fig.tight_layout()


#%%
'''
Fig 2E, we compare PC2 with HV and PC5 with AO, here we get the most similar maps.
This time we change it into horizontal stacks.
'''
hv_resp = ac.Orien_t_graphs['H-V'].loc['CohenD',:]
hv_map = ac.Generate_Weighted_Cell(hv_resp)
ao_resp = ac.Orien_t_graphs['A-O'].loc['CohenD',:]
ao_map = ac.Generate_Weighted_Cell(ao_resp)

# plot pca parts
value_max = 0.1
value_min = -0.1
plt.clf()
plt.cla()
fig,axes = plt.subplots(nrows=1, ncols=2,figsize = (7,4),dpi = 180)
# cbar_ax = fig.add_axes([1, .45, .02, .2])
sns.heatmap(ac.Generate_Weighted_Cell(spon_pcs[1,:]),center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = value_max,vmin = value_min,cbar=False,square=True,cmap = 'gist_gray')
sns.heatmap(ac.Generate_Weighted_Cell(spon_pcs[4,:]),center = 0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = value_max,vmin = value_min,cbar=False,square=True,cmap = 'gist_gray')
# axes[0].set_title('PC2',size = 14)
# axes[1].set_title('PC5',size = 14)
fig.tight_layout()

#%% and plot functional map parts
value_max = 4
value_min = -4
plt.clf()
plt.cla()
fig,axes = plt.subplots(nrows=1, ncols=2,figsize = (7,4),dpi = 180)
# cbar_ax = fig.add_axes([1, .45, .02, .2])
sns.heatmap(-hv_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = value_max,vmin = value_min,cbar = False,square=True)
sns.heatmap(ao_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = value_max,vmin = value_min,cbar = False,square=True)
# axes[0].set_title('90°-0°',size = 14)
# axes[1].set_title('45°-135°',size = 14)
fig.tight_layout()

hv_r,_ = stats.pearsonr(spon_pcs[1,:],-hv_resp)
ao_r,_ = stats.pearsonr(spon_pcs[4,:],ao_resp)
print(f'HV Corr:{hv_r:.3f};AO Corr:{ao_r:.3f};')

#%%
'''
Figs 2F, Correlation of PC comps with shuffle This will be done on stats file.
'''





#%%
'''
Figs 2G, we compare top 10 PCs with all 6 functional maps, and we get similars.
Figs S2G is the same, just change spon_pcs into spon_pcs_s
'''
# get all response
od_resp = ac.OD_t_graphs['OD'].loc['CohenD',:]
hv_resp = ac.Orien_t_graphs['H-V'].loc['CohenD',:]
ao_resp = ac.Orien_t_graphs['A-O'].loc['CohenD',:]
red_resp = ac.Color_t_graphs['Red-White'].loc['CohenD',:]
blue_resp = ac.Color_t_graphs['Blue-White'].loc['CohenD',:]
all_response = [od_resp,hv_resp,ao_resp,red_resp,blue_resp]

#and generate data frame
pc_list = ['PC{}'.format(i) for i in range(1, 11)]
networks = ['OD','HV','AO','Red','Blue']
all_corrs = pd.DataFrame(0.0,columns = pc_list,index = networks)

# fill it with pearsonr
for i,c_pc in enumerate(pc_list):
    c_pc_response = spon_pcs[i,:]
    for j,c_net in enumerate(networks):
        c_stim_response = all_response[j]
        c_r,_ = stats.pearsonr(c_pc_response,c_stim_response)
        all_corrs.loc[c_net,c_pc] = abs(c_r)

#%% Plot it.
value_max = 0.8
value_min = 0
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (5,8),dpi = 300)
sns.heatmap(all_corrs.T,center = 0,annot=True,cmap = 'bwr', fmt=".2f",ax = ax,vmax = value_max,vmin = value_min,cbar=False,annot_kws={"size": 14})

# ax.set_xticks([])
# ax.set_yticks([])
ax.set_yticks(np.arange(0,10)+0.5)
ax.set_yticklabels(np.arange(1,11),fontsize = 14)
ax.set_xticklabels(['OD','0°-90°','45°-135°','Red','Blue'],fontsize = 14)
ax.xaxis.tick_top()
plt.show()

