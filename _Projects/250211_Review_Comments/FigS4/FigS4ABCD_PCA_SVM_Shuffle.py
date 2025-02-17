'''
The same graph as for fig 2, but on shuffled graph.

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
savepath = r'G:\我的云端硬盘\#Figs\#250211_Revision1\FigS4'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
sponrun = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
start = sponrun.index[0]
end = sponrun.index[-1]
spon_series = Z_refilter(ac,'1-001',start,end).T
spon_series = Spon_Shuffler(spon_series,filter_para=(0.005,0.65))
#%%
'''
Fig S4 ACD, PCA on shuffled series.
'''
pcnum=10

spon_pcs,spon_coords,spon_model = Z_PCA(Z_frame=spon_series,sample='Frame',pcnum=pcnum)

#%% Fig S4 A, PCA embedding
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
fig.savefig(ot.join(savepath,'S2A_Shuffled_Spon_PCA.png'),bbox_inches = 'tight')

#%% Fig S4C, top 10 PCs
plt.clf()
plt.cla()

vmax = 0.1
vmin = -0.1

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
fig.savefig(ot.join(savepath,'S2D_Shuffled_PCs.png'),bbox_inches='tight')

#%% Fig S4D, best corr with funcmap, horizontal.
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
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (9,5),dpi = 300)
sns.heatmap(all_corrs,center = 0,annot=True,cmap = 'bwr', fmt=".2f",ax = ax,vmax = value_max,vmin = value_min,cbar=False,annot_kws={"size": 12})

# ax.set_xticks([])
# ax.set_yticks([])
ax.set_xticks(np.arange(0,10)+0.5)
ax.set_xticklabels(np.arange(1,11),fontsize = 12)
ax.set_yticklabels(['OD','0°-90°','45°-135°','Red','Blue'],fontsize = 12)
# ax.xaxis.tick_top()
# g.xaxis.set_ticks_position("top")
fig.savefig(ot.join(savepath,'S2C_All_PC_Corr.png'),bbox_inches='tight')
plt.show()
#%%
'''
Fig S4B, stats of explained var for shuffle.
'''

all_pc_var_s,all_pc_best_corr_s,pc1_corrs_s = ot.Load_Variable(r'G:\我的云端硬盘\#Figs\#250211_Revision1\Fig2','All_PC_Corr_Infos_Shuffle500.pkl')

plotable = pd.DataFrame(all_pc_var_s.T).melt(var_name='PC',value_name='Explained VAR Ratio')
plotable['Explained VAR Ratio'] = plotable['Explained VAR Ratio']*100
plotable['PC'] = plotable['PC']+1

plt.clf()
plt.cla()
fontsize = 14

fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (5,4),dpi = 300)
sns.barplot(data = plotable,y = 'Explained VAR Ratio',x = 'PC',ax = ax,capsize=0.2)
# ax.set_xlabel('PC',size = 12)
# ax.set_ylabel('Explained Ratio(%)',size = 12)
# ax.set_title('Each PC explained Variance',size = 14)
ax.set_ylim(0,32)
top10_sum = all_pc_var_s.sum(0)
ax.set_yticks([0,10,20,30])
ax.set_yticklabels([0,10,20,30],fontsize = fontsize)
ax.set_xticks(np.arange(0,10))
ax.set_xticklabels(np.arange(1,11),fontsize = fontsize)

ax.set_ylabel('')
ax.set_xlabel('')
print(f'Top 10 PC explain VAR={top10_sum.mean():.4f}, std={top10_sum.std():.4f}')
fig.savefig(ot.join(savepath,'S2B_Shuffled_Spon_VAR.png'), bbox_inches='tight')