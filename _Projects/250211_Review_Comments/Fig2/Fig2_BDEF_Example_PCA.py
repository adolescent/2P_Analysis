'''
This script shows example of PCA analysis for spontaneous response.
The same code, except for removing the low pass filter.
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

expt_folder = r'D:\#Fig_Data\_All_Spon_Data_V1\L76_18M_220902'
savepath = r'D:\_GoogleDrive_Files\#Figs\#250211_Revision1\Fig2'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
sponrun = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
start = sponrun.index[0]
end = sponrun.index[-1]
spon_series = Z_refilter(ac,'1-001',start,end).T

#%%
'''
Fig2B, we do pca on given spon matrix, and we show embedding of pc1-3.
2B-2 will show the explained var of given location.
'''

spon_series = np.array(spon_series)
pcnum = 10
# real spon models
spon_pcs,spon_coords,spon_model = Z_PCA(Z_frame=spon_series,sample='Frame',pcnum=pcnum)
model_var_ratio = np.array(spon_model.explained_variance_ratio_)
print(f'{pcnum} PCs explain Spontaneous VAR {model_var_ratio[:pcnum].sum()*100:.1f}%')

#%% Plot raw embeddings of previous 3 PCs.
u = spon_coords[:,:3]

plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (6,6),dpi = 300,subplot_kw=dict(projection='3d'))
orien_elev = 25
orien_azim = 50

ax.grid(False)
ax.view_init(elev=orien_elev, azim=orien_azim)
ax.set_box_aspect(aspect=None, zoom=1) # shrink graphs
ax.axes.set_xlim3d(left=-40, right=40)
ax.axes.set_ylim3d(bottom=-30, top=30)
ax.axes.set_zlim3d(bottom=30, top=-30)
ax.scatter3D(u[:,0],u[:,1],u[:,2],s = 5,c = [0.7,0.7,0.7],alpha = 1,lw = 0)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
fig.tight_layout()

fig.savefig(ot.join(savepath,'Fig2B_Spon_Embedding.png'),bbox_inches='tight')
#%%
'''
Fig 2D, generate PCA's main axis, compare it with stimulus map.
'''
vmax = 0.1
vmin = -0.1
# plot bar
plt.clf()
plt.cla()
fig_bar,ax_bar = Cbar_Generate(vmin=vmin,vmax=vmax,cmap='gist_gray',aspect=7,labelsize=8)
# fig_bar.savefig(ot.join(savepath,'Fig2D_Bars.png'),bbox_inches='tight')
#%% plot top 10 pc
plt.clf()
plt.cla()
font_size = 13
fig,axes = plt.subplots(nrows=2, ncols=5,figsize = (12,6),dpi = 300)
# cbar_ax = fig.add_axes([1, .45, .01, .2])
for i in tqdm(range(10)):
    c_pc = spon_pcs[i,:]
    c_pc_graph = ac.Generate_Weighted_Cell(c_pc)
    # sns.heatmap(c_pc_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[i//5,i%5],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True,cmap = cmaps.pinkgreen_light)
    sns.heatmap(c_pc_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[i//5,i%5],vmax = vmax,vmin = vmin,cbar= False,square=True,cmap = 'gist_gray')
    # axes[i//5,i%5].set_title(f'PC {i+1}',size = font_size)

fig.tight_layout()
fig.savefig(ot.join(savepath,'Fig2D_PC_Comps.png'),bbox_inches='tight')
#%%
'''
Fig 2E, compare correlation between func maps and top 10 PCs
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

fig.savefig(ot.join(savepath,'Fig2E_All_PC_Corr.png'),bbox_inches='tight')
plt.show()
#%%
'''
Fig 2F, example of PC3,4 and HV-AO graph.

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
sns.heatmap(ac.Generate_Weighted_Cell(spon_pcs[2,:]),center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = value_max,vmin = value_min,cbar=False,square=True,cmap = 'gist_gray')
sns.heatmap(ac.Generate_Weighted_Cell(spon_pcs[3,:]),center = 0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = value_max,vmin = value_min,cbar=False,square=True,cmap = 'gist_gray')
# axes[0].set_title('PC3',size = 14)
# axes[1].set_title('PC4',size = 14)
fig.tight_layout()
fig.savefig(ot.join(savepath,'Fig2Fa_PC34.png'),bbox_inches='tight')

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
fig.savefig(ot.join(savepath,'Fig2Fb_VH-AO.png'),bbox_inches='tight')

hv_r,_ = stats.pearsonr(spon_pcs[2,:],-hv_resp)
ao_r,_ = stats.pearsonr(spon_pcs[3,:],ao_resp)
print(f'HV Corr:{hv_r:.3f};AO Corr:{ao_r:.3f};')
#%% Fig2Fb Bars
vmax = 4
vmin = -4
# plot bar
plt.clf()
plt.cla()
fig_bar,ax_bar = Cbar_Generate(vmin=vmin,vmax=vmax,cmap=None,aspect=7,labelsize=8)
# fig_bar.savefig(ot.join(savepath,'Fig2D_Bars.png'),bbox_inches='tight')

