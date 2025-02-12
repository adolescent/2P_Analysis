
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
# import umap
# import umap.plot
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
from Cell_Class.Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *



exp_path = r'D:\_All_Spon_Data_V1\L76_SM_Run03bug_210721'
# exp_path = r'D:\_All_Spon_Data_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(exp_path,'Cell_Class.pkl')
sponrun =  ot.Load_Variable_v2(exp_path,'Spon_Before.pkl')
start = sponrun.index[0]
end = sponrun.index[-1]
all_cell_dic = ac.all_cell_dic
fps = ac.fps

savepath = r'D:\ZR\_Data_Temp\_Article_Data\_Revised_Data'

from Fix_Funcs import *
z_lp_on = Z_refilter(ac,start,end,'1-001',0.005,0.3).T
z_lp_off = Z_refilter(ac,start,end,'1-001',0.005,False).T

#%% #########################################
'''
Part 1, test PCA on unfilted results.
'''
pcnum = 10

spon_pcs,spon_coords,spon_model = Z_PCA(Z_frame=z_lp_on,sample='Frame',pcnum=pcnum)
model_var_ratio = np.array(spon_model.explained_variance_ratio_)
print(f'{pcnum} PCs explain Spontaneous VAR {model_var_ratio[:pcnum].sum()*100:.1f}%')

spon_pcs_off,spon_coords_off,spon_model_off = Z_PCA(Z_frame=z_lp_off,sample='Frame',pcnum=pcnum)
model_var_ratio_off = np.array(spon_model_off.explained_variance_ratio_)
print(f'{pcnum} PCs explain Spontaneous VAR {model_var_ratio_off[:pcnum].sum()*100:.1f}%')

#%% Plot PCA explained VARs.
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (6,4),dpi = 144)
sns.barplot(y = model_var_ratio*100,x = np.arange(1,11),ax = ax,alpha = 0.7)
sns.barplot(y = model_var_ratio_off*100,x = np.arange(1,11),ax = ax,alpha = 0.7)

ax.set_xlabel('PC',size = 12)
ax.set_ylabel('Explained Variance (%)',size = 12)
ax.set_title('Each PC explained Variance',size = 14)
ax.set_ylim(0,37)

#%%
'''
Fig 2c & S2c, we will generate PCA main axis, and compare it with stimulus maps, getting the most similar PC between stimulus and spon.
'''
# only PC graph, it's very easy.
# import cmasher as cmr
# import colormaps as cmaps

plt.clf()
plt.cla()
value_max = 0.1
value_min = -0.1
font_size = 13
fig,axes = plt.subplots(nrows=2, ncols=5,figsize = (12,6),dpi = 300)
# cbar_ax = fig.add_axes([1, .45, .01, .2])
for i in tqdm(range(10)):
    c_pc = spon_pcs_off[i,:]
    c_pc_graph = ac.Generate_Weighted_Cell(c_pc)
    # sns.heatmap(c_pc_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[i//5,i%5],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True,cmap = cmaps.pinkgreen_light)
    sns.heatmap(c_pc_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[i//5,i%5],vmax = value_max,vmin = value_min,cbar= False,square=True,cmap = 'gist_gray')
    # axes[i//5,i%5].set_title(f'PC {i+1}',size = font_size)

fig.tight_layout()


#%% 

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

# and plot it
value_max = 0.8
value_min = 0
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (4,7),dpi = 180)
sns.heatmap(all_corrs.T,center = 0,annot=True, fmt=".3f",ax = ax,vmax = value_max,vmin = value_min,cbar=False,cmap = 'bwr')
plt.show()

