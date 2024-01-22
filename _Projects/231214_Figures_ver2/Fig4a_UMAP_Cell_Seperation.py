'''
This script will work on generating cell seperated graph.

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
from Cell_Class.Timecourse_Analyzer import *

work_path = r'D:\_Path_For_Figs\2401_Amendments\Fig3_New'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
# some times we need to ignore warnings.
import warnings
warnings.filterwarnings("ignore")

#%% ################ 1. GENERATE PAIR CORR MATRIX.###################
all_best_oriens = ot.Load_Variable(work_path,'All_Cell_Best_Oriens.pkl')
all_cell_corr = {}
for i,cloc in enumerate(all_path_dic): # test 1 location.
    cloc_name = cloc.split('\\')[-1]
    c_best_orien = all_best_oriens[cloc_name]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    c_tuned_cells_orien = c_best_orien[c_best_orien['Tuned']==1]
    c_tuned_cells = list(c_tuned_cells_orien.index)
    pairnum = int(len(c_tuned_cells)*(len(c_tuned_cells)-1)/2)
    cloc_corr_frame = pd.DataFrame(0,range(pairnum),columns = ['Corr','CellA','CellB','DistX','DistY','OD_A','OD_B','OrienA','OrienB','Dist','OD_Diff','Orien_Diff'])
    counter = 0
    cloc_OD = ac.OD_t_graphs['OD'].loc['CohenD']
    for j in tqdm(range(len(c_tuned_cells))):
        cell_A = c_tuned_cells[j]
        cell_A_coords = ac.Cell_Locs[cell_A]
        spon_A = np.array(c_spon.loc[:,cell_A])
        od_A = cloc_OD[cell_A]
        best_orien_A = c_tuned_cells_orien.loc[cell_A,'Best_Angle']
        for k in range(j+1,len(c_tuned_cells)):
            cell_B = c_tuned_cells[k]
            cell_B_coords = ac.Cell_Locs[cell_B]
            spon_B = np.array(c_spon.loc[:,cell_B])
            od_B = cloc_OD[cell_B]
            best_orien_B = c_tuned_cells_orien.loc[cell_B,'Best_Angle']
            # calculate difference,
            c_corr,_ = stats.pearsonr(spon_A,spon_B)
            c_distx = cell_A_coords['X']-cell_B_coords['X']
            c_disty = cell_A_coords['Y']-cell_B_coords['Y']
            c_od_diff = abs(od_A-od_B)
            c_dist = np.sqrt(c_distx**2+c_disty**2)
            c_orien_diff = abs(best_orien_A-best_orien_B)
            c_orien_diff = min(c_orien_diff,180-c_orien_diff)
            cloc_corr_frame.loc[counter,:] = [c_corr,cell_A,cell_B,c_distx,c_disty,od_A,od_B,best_orien_A,best_orien_B,c_dist,c_od_diff,c_orien_diff]
            counter += 1
    all_cell_corr[cloc_name] = cloc_corr_frame
ot.Save_Variable(work_path,'All_Pair_Corrs',all_cell_corr)
#%% ################################# 2. PLOT CORR MAPS #########################################
all_locname = list(all_cell_corr.keys())
example_loc = all_cell_corr[all_locname[4]]
example_loc['OrienA_bin'] = pd.cut(example_loc['OrienA'], bins=np.linspace(0,180,31), right=False)
example_loc['OrienB_bin'] = pd.cut(example_loc['OrienB'], bins=np.linspace(0,180,31), right=False)
# OD is a little fuzzy, as we need to make +- part have similar bins.

half_bin = 15 # make +- have same bin
od_min = example_loc['OD_B'].min()
# od_min = -2.8
od_max = example_loc['OD_B'].max()-0.2
# od_max = 1.2
od_bins = np.concatenate((np.linspace(od_min,0,half_bin,endpoint=False),(np.linspace(0,od_max,half_bin))))
example_loc['OD_A_bin'] = pd.cut(example_loc['OD_A'], bins=od_bins, right=False)
example_loc['OD_B_bin'] = pd.cut(example_loc['OD_B'], bins=od_bins, right=False)

# Dist neet to use abs 
dist_bins = np.linspace(0,450,31)
example_loc['DistX_bin'] = pd.cut(abs(example_loc['DistX']), bins=dist_bins, right=False)
example_loc['DistY_bin'] = pd.cut(abs(example_loc['DistY']), bins=dist_bins, right=False)

# extend groups to get symetry matrix.
example_loc_sym = copy.deepcopy(example_loc)
temp =  copy.deepcopy(example_loc_sym['OD_B_bin'])
example_loc_sym['OD_B_bin'] = example_loc_sym['OD_A_bin']
example_loc_sym['OD_A_bin'] = temp
temp =  copy.deepcopy(example_loc_sym['OrienB_bin'])
example_loc_sym['OrienB_bin'] = example_loc_sym['OrienA_bin']
example_loc_sym['OrienA_bin'] = temp
example_loc = pd.concat([example_loc,example_loc_sym])

orien_plotable = example_loc.groupby(['OrienA_bin', 'OrienB_bin'], as_index=False)['Corr'].mean()
od_plotable = example_loc.groupby(['OD_A_bin', 'OD_B_bin'], as_index=False)['Corr'].mean()
dist_plotable = example_loc.groupby(['DistX_bin','DistY_bin'], as_index=False)['Corr'].mean()
# orien_size = example_loc.groupby(['OrienA_bin', 'OrienB_bin'], as_index=False)['Corr'].size()
# od_size = example_loc.groupby(['OD_A_bin', 'OD_B_bin'], as_index=False)['Corr'].size()
# 1. Plot X-Y Orien pref maps.
# seperate x and y by oriens. In 
plt.clf()
plt.cla()
center = 0.28
vmax = 0.5
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(17,6),dpi = 180)
cbar_ax = fig.add_axes([.92, .2, .02, .6])
heatmap_data_orien = orien_plotable.pivot(index='OrienA_bin', columns='OrienB_bin', values='Corr')
heatmap_data_od = od_plotable.pivot(index='OD_A_bin', columns='OD_B_bin', values='Corr')
heatmap_data_dist = dist_plotable.pivot(index='DistX_bin', columns='DistY_bin', values='Corr')
for i,c_map in enumerate([heatmap_data_od,heatmap_data_orien,heatmap_data_dist]):
    c_map.columns = [str(bin.left) for bin in c_map.columns]
    c_map.index = [str(bin.left) for bin in c_map.index]

g1 = sns.heatmap(heatmap_data_dist, cbar=True,center =center,vmax = vmax,square= True,ax = axes[0],cbar_ax=cbar_ax,xticklabels=False,yticklabels=False)
g2 = sns.heatmap(heatmap_data_od, cbar=True,center =center,vmax = vmax,square= True,ax = axes[1],cbar_ax=cbar_ax,xticklabels=False,yticklabels=False)
g3 = sns.heatmap(heatmap_data_orien, cbar=True,center =center,vmax = vmax,square= True,ax = axes[2],cbar_ax=cbar_ax,xticklabels=False,yticklabels=False)
g1.set_facecolor('gray')
g2.set_facecolor('gray')
g3.set_facecolor('gray')
# sns.heatmap(heatmap_data, cbar=True,square= True,ax = ax,cbar_ax=cbar_ax)
## legends here.
# orientation
axes[0].set_xlabel('Distance X')
axes[0].set_ylabel('Distance Y')
axes[0].set_title('Correlation vs Distance')
axes[0].set_xticks([0,6,12,18,24,30])
axes[0].set_xticklabels([0,90,180,270,360,450])
axes[0].set_yticks([0,6,12,18,24,30])
axes[0].set_yticklabels([0,90,180,270,360,450])
# od
axes[1].set_xlabel('OD Tuning A')
axes[1].set_ylabel('OD Tuning B')
axes[1].set_title('Correlation vs OD Tuning')
# od_ticks = [0,7,11,15,20,24,29]
od_ticks = [0,5,10,15,20,24,29]
od_ticks_label = []
for i,c_group in enumerate(od_ticks):
    od_ticks_label.append(np.round(od_bins[c_group],2))
axes[1].set_xticks(od_ticks)
axes[1].set_xticklabels(od_ticks_label)
axes[1].set_yticks(od_ticks)
axes[1].set_yticklabels(od_ticks_label)

orien_ticks = [0,5,10,15,20,25,30]
orien_ticks_label = []
for i,c_group in enumerate(orien_ticks):
    orien_ticks_label.append(c_group*6)
axes[2].set_xticks(orien_ticks)
axes[2].set_xticklabels(orien_ticks_label)
axes[2].set_yticks(orien_ticks)
axes[2].set_yticklabels(orien_ticks_label)
axes[2].set_xlabel('Orientation A')
axes[2].set_ylabel('Orientation B')
axes[2].set_title('Correlation vs Orientation Tuning')
# fig.tight_layout()
for i in range(3):
    # axes[i].xaxis.set_label_position('top') 
    axes[i].xaxis.tick_top()

plt.show()



