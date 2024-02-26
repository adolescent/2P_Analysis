'''
This will generate all cell series heat map, and average of given time window.
'''
#%%
from Cell_Class.Stim_Calculators import Stim_Cells
from Cell_Class.Format_Cell import Cell
import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Plotter.Line_Plotter import EZLine
from tqdm import tqdm
import cv2
import re

work_path = r'D:\_Path_For_Figs\_2312_ver2\Fig1'
expt_folder = r'D:\_All_Spon_Datas_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
ac.wp = expt_folder
global_avr = cv2.imread(r'D:\_All_Spon_Datas_V1\L76_18M_220902\Global_Average_cai.tif',0) # read as 8 bit gray scale map.
#%%##################### FIG 1F, HEAT MAP ########################################
# Step1, sort cell by orientation preference.
spon_series = ot.Load_Variable(expt_folder,'Spon_Before.pkl').reset_index(drop = True)
orien_series = ac.Z_Frames['1-007']
rank_index = pd.DataFrame(index = ac.acn,columns=['Best_Orien','Sort_Index','Sort_Index2'])
for i,cc in enumerate(ac.acn):
    rank_index.loc[cc]['Best_Orien'] = ac.all_cell_tunings[cc]['Best_Orien']
    if ac.all_cell_tunings[cc]['Best_Orien'] == 'False':
        rank_index.loc[cc]['Sort_Index']=-1
        rank_index.loc[cc]['Sort_Index2']=0
    else:
        orien_tunings = float(ac.all_cell_tunings[cc]['Best_Orien'][5:])
        # rank_index.loc[cc]['Sort_Index'] = np.sin(np.deg2rad(orien_tunings))
        rank_index.loc[cc]['Sort_Index'] = orien_tunings
        rank_index.loc[cc]['Sort_Index2'] = np.cos(np.deg2rad(orien_tunings))
# actually we sort only by raw data.
sorted_cell_sequence = rank_index.sort_values(by=['Sort_Index'],ascending=False)
# and we try to reindex data.
sorted_stim_response = orien_series.T.reindex(sorted_cell_sequence.index)
sorted_spon_response = spon_series.T.reindex(sorted_cell_sequence.index)
#%% Step2, Plot heat map with square for average.
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 4),dpi = 180)
cbar_ax = fig.add_axes([.92, .2, .02, .6])
# plot heat maps, and reset frame index.
sns.heatmap((sorted_stim_response.iloc[:,1000:1500].T.reset_index(drop = True).T),center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = 6,vmin = -2,cbar_ax= cbar_ax)
sns.heatmap(sorted_spon_response.iloc[:,2000:2500].T.reset_index(drop = True).T,center = 0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = 6,vmin = -2,cbar_ax= cbar_ax)
# set ticks and ticks label.
xticks = np.array([0,100,200,300,400,500])
axes[1].set_xticks(xticks)
axes[1].set_xticklabels([0,100,200,300,400,500])
# Add selected location for compare.
from matplotlib.patches import Rectangle
axes[0].add_patch(Rectangle((175,0), 6, 520, fill=False, edgecolor='yellow', lw=1,alpha = 0.8))
axes[1].add_patch(Rectangle((46,0), 6, 520, fill=False, edgecolor='yellow', lw=1,alpha = 0.8))
# set graph titles
# fig.tight_layout(rect=[0, 0, .9, 1])
axes[0].set_title('Stim-induced Response')
axes[1].set_title('Spontaneous Response')
axes[1].set_xlabel(f'Frames')
axes[1].set_ylabel(f'Cells')
axes[0].set_ylabel(f'Cells')
# axes[1].xaxis.set_visible(True)
# axes[1].set_xticks([0,100,200,300,400,500])
plt.show()
#%% Step3, average points above and compare stim and spon.
stim_start_point = 175
spon_start_point = 46
stim_recover = orien_series.loc[1000+stim_start_point:1000+stim_start_point+6].mean(0)
spon_recover = spon_series.loc[2000+spon_start_point:2000+spon_start_point+6].mean(0)
stim_recover_map = ac.Generate_Weighted_Cell(stim_recover)
spon_recover_map = ac.Generate_Weighted_Cell(spon_recover)

plt.clf()
plt.cla()
vmax = 4
vmin = -3
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4,6),dpi = 180)
fig.tight_layout()
cbar_ax = fig.add_axes([.9, .3, .05, .4])

sns.heatmap(stim_recover_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax,square=True)
sns.heatmap(spon_recover_map,center=0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax,square=True)
axes[0].set_title('Stim-induced Response')
axes[1].set_title('Spontaneous Response')
plt.show()
