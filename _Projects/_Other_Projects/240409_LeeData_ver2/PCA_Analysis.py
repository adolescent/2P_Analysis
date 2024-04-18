'''

Do PCA on raw data and binned data.

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
from Filters import Signal_Filter_v2
from Cell_Class.Advanced_Tools import *

expt_folder = r'D:\_Lee_Data\231219_Lee_Data_31fps'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class_Raw.pkl')# ac class is different from our own data!

import warnings
warnings.filterwarnings("ignore")
all_series = ot.Load_Variable(expt_folder,'4bin_Frames.pkl')

#%% 


pcnum = 120

used_data = all_series['Z_score']
# used_data = np.reshape(all_series['Z_score'],(6925,4,332)).mean(1)
spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=used_data,sample='Frame',pcnum=pcnum)
model_var_ratio = np.array(spon_models.explained_variance_ratio_)
print(f'{pcnum} PCs explain Spontaneous VAR {model_var_ratio[:pcnum].sum()*100:.1f}%')

#%% Plot PC explained variance here.
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (6,4),dpi = 144)
sns.barplot(y = model_var_ratio*100,x = range(120),ax = ax)
ax.set_xlabel('PC',size = 12)
ax.set_ylabel('Explained Variance (%)',size = 12)
ax.set_xticks('')
ax.set_title('Each PC explained Variance',size = 14)

#%% Recover PCs.
all_pcs_graph = np.zeros(shape = (pcnum,512,512))
clip_std = 5
for i in tqdm(range(pcnum)):
    c_pc = spon_pcs[i,:]
    c_graph = ac.Generate_Weighted_Cell(c_pc)
    c_graph = np.clip(c_graph,-c_graph.std()*clip_std,c_graph.std()*clip_std)
    all_pcs_graph[i,:,:] = c_graph

#%% Plot PCs
plotable_graphs = all_pcs_graph[:15,:,:]
plt.clf()
plt.cla()
value_max = 0.1
value_min = -0.1
font_size = 16
fig,axes = plt.subplots(nrows=3, ncols = 5,figsize = (14,8),dpi = 180)
cbar_ax = fig.add_axes([1, .4, .01, .2])
for i in range(15):
    sns.heatmap(plotable_graphs[i,:,:],center = 0,xticklabels=False,yticklabels=False,ax = axes[i//5,i%5],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True,cbar_kws={'label': 'PC Weight'})
fig.tight_layout()