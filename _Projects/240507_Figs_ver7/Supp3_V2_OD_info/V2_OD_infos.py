'''
This script will try to get OD infos for V2 data points

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
from Classifier_Analyzer import *

import warnings
warnings.filterwarnings("ignore")



datapath = r'D:\_All_Spon_Data_V2'
savepath = r'D:\_Path_For_Figs\230507_Figs_v7\V2_ODs'
all_path_dic = list(ot.Get_Subfolders(datapath))
all_path_dic.pop(0)
#%% Get each OD graphs.

all_od_index = pd.DataFrame(columns = ['Loc','Cell','OD_index','LE','RE'])
all_loc_graphs = {}
for i,cloc in enumerate(all_path_dic):
    cloc_name = cloc.split('\\')[-1]
    c_ac = ot.Load_Variable(cloc,'Cell_Class.pkl')
    c_LE_on = c_ac.OD_t_graphs['L-0'].loc['A_reponse',:]
    c_RE_on = c_ac.OD_t_graphs['R-0'].loc['A_reponse',:]
    c_diff = c_LE_on-c_RE_on
    c_od_index = (c_LE_on-c_RE_on)/(c_LE_on+c_RE_on)

    # get LE,RE and LE-RE maps.
    c_LE_frame = c_ac.Generate_Weighted_Cell(c_LE_on)
    c_RE_frame = c_ac.Generate_Weighted_Cell(c_RE_on)
    c_OD_frame = c_ac.Generate_Weighted_Cell(c_diff)
    all_loc_graphs[cloc_name] = {}
    all_loc_graphs[cloc_name]['LE'] = c_LE_frame
    all_loc_graphs[cloc_name]['RE'] = c_RE_frame
    all_loc_graphs[cloc_name]['OD'] = c_OD_frame
    
    # and get all cell response
    for j in range(len(c_od_index)):
        all_od_index.loc[len(all_od_index),:] = [cloc_name,j+1,c_od_index[j+1],c_LE_on[j+1],c_RE_on[j+1]]


#%% Plot OD Graphs

all_locs = list(all_loc_graphs.keys())
c_graphs = all_loc_graphs[all_locs[1]]

plt.clf()
plt.cla()

vmax = 1.5
vmin = -1.5

fig,axes = plt.subplots(nrows=1, ncols=3,figsize = (7,5),dpi = 180)
cbar_ax = fig.add_axes([1.0, .4, .015, .2])
sns.heatmap(c_graphs['LE'],center = 0,square = True,vmax = vmax,vmin = vmin,cbar_ax=cbar_ax,xticklabels=False,yticklabels=False,ax = axes[0])
sns.heatmap(c_graphs['RE'],center = 0,square = True,vmax = vmax,vmin = vmin,cbar_ax=cbar_ax,xticklabels=False,yticklabels=False,ax = axes[1])
sns.heatmap(c_graphs['OD'],center = 0,square = True,vmax = vmax,vmin = vmin,cbar_ax=cbar_ax,xticklabels=False,yticklabels=False,ax = axes[2])

axes[0].set_title('LE')
axes[1].set_title('RE')
axes[2].set_title('LE-RE')
fig.tight_layout()

#%% Plot OD infos.
plt.clf()
plt.cla()

fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (7,5),dpi = 180)

med = all_od_index['OD_index'].median()
ax.axvline(x = med,linestyle = '--',color = 'gray')
sns.histplot(data = all_od_index,x = 'OD_index',ax = ax,bins = np.linspace(-1,1,40))
ax.set_xlabel('OD Index')
