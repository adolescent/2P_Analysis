'''
This script will generate stim, both Z t map,dff map and df map are generated.

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

work_path = r'D:\_Path_For_Figs\240228_Figs_v4\Fig1'
expt_folder = r'D:\_All_Spon_Data_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
ac.wp = expt_folder
# global_avr = cv2.imread(r'D:\_All_Spon_Datas_V1\L76_18M_220902\Global_Average_cai.tif',0) # read as 8 bit gray scale map.
# cell_example_list = [47,322,338]
raw_orien_run = ot.Load_Variable(f'{expt_folder}\\Orien_Frames_Raw.pkl')
raw_od_run = ot.Load_Variable(f'{expt_folder}\\OD_Frames_Raw.pkl')
raw_color_run = ot.Load_Variable(f'{expt_folder}\\Color_Frames_Raw.pkl')
#%% #############################FIG 1D VER 1, ORIGIONAL T MAP###################
od_graph = ac.Generate_Weighted_Cell(ac.OD_t_graphs['OD'].loc['t_value'])
ao_graph = ac.Generate_Weighted_Cell(ac.Orien_t_graphs['A-O'].loc['t_value'])
red_graph = ac.Generate_Weighted_Cell(ac.Color_t_graphs['Red-White'].loc['t_value'])
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 4))
value_max = 50
value_min = -40
cbar_ax = fig.add_axes([.91, .15, .015, .7])

sns.heatmap(od_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[0],square=True,cbar_ax= cbar_ax,vmax = value_max,vmin = value_min)
sns.heatmap(ao_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[1],square=True,cbar_ax= cbar_ax,vmax = value_max,vmin = value_min)
sns.heatmap(red_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[2],square=True,cbar_ax= cbar_ax,vmax = value_max,vmin = value_min)

axes[0].set_title('Left Eye - Right Eye',size = 16)
axes[1].set_title('Orientation 45 - Orientation 135',size = 16)
axes[2].set_title('Red Color - White Color',size = 16)

fig.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=0.88, top=None, hspace=None)
plt.show()


#%%#################################FIG 1D VER2, RAW F MAP####################################

# orien first.
orien_avr = raw_orien_run.mean(0)
n_std = 3
a_ids = ac.Stim_Frame_Align['Run007'][11]
b_ids = ac.Stim_Frame_Align['Run007'][-1]
c_ids = ac.Stim_Frame_Align['Run007'][15]

def Map_Grabber(raw_frame,ids,clip_std):
    on_frames = raw_frame[ids,:,:].mean(0)
    on_max = min(on_frames.flatten().max(),(on_frames.flatten().mean()+clip_std*on_frames.flatten().std()))
    clipped_frame = np.clip(on_frames,0,on_max)
    return clipped_frame

a_on = Map_Grabber(raw_orien_run,a_ids,n_std)
b_on = Map_Grabber(raw_orien_run,b_ids,n_std)
c_on = Map_Grabber(raw_orien_run,c_ids,n_std)
norm_index = max(a_on.max(),b_on.max(),c_on.max())
a_graph = (a_on*65535/norm_index).astype('u2')
b_graph = (b_on*65535/norm_index).astype('u2')
c_graph = (c_on*65535/norm_index).astype('u2')
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8,4))
axes[0].imshow(a_graph,cmap='gray', vmin=0, vmax=65535)
axes[1].imshow(b_graph,cmap='gray', vmin=0, vmax=65535)
axes[2].imshow(c_graph,cmap='gray', vmin=0, vmax=65535)
axes[0].set_title('Orien 45 Stim ON')
axes[1].set_title('ISI')
axes[2].set_title('Orien 135 Stim ON')

for i in range(3):
    axes[i].axis('off')

#%% ##########################FIG 1D VER2, RAW dF MAP################################## 
'''
This part only subtract graph from each other, get subtraction dF maps.
'''
# orien first
orien_avr = raw_orien_run.mean(0)
n_std = 5
orien45_ids = ac.Stim_Frame_Align['Run007'][11]
orien135_ids = ac.Stim_Frame_Align['Run007'][15]
# b_ids = ac.Stim_Frame_Align['Run007'][-1]


def Graph_Subtraction(raw_frame,a_ids,b_ids,clip_std):
    a_frames = raw_frame[a_ids,:,:].mean(0)
    b_frames = raw_frame[b_ids,:,:].mean(0)
    sub_frame = a_frames-b_frames
    raw_sub_graph= np.clip(sub_frame,(sub_frame.mean()-clip_std*sub_frame.std()),(sub_frame.mean()+clip_std*sub_frame.std()))
    return raw_sub_graph
ao_subgraph = Graph_Subtraction(raw_orien_run,orien45_ids,orien135_ids,n_std)
# then OD
LE_ids = []
for i,c_stim in enumerate([1,3,5,7]):
    LE_ids.extend(ac.Stim_Frame_Align['Run006'][c_stim])
RE_ids = []
for i,c_stim in enumerate([2,4,6,8]):
    RE_ids.extend(ac.Stim_Frame_Align['Run006'][c_stim])
od_subgraph = Graph_Subtraction(raw_od_run,LE_ids,RE_ids,n_std)
# Then Red-White
red_ids = []
for i,c_stim in enumerate([1, 8, 15, 22]):
    red_ids.extend(ac.Stim_Frame_Align['Run008'][c_stim])
white_ids = []
for i,c_stim in enumerate([7, 14, 21, 28]):
    white_ids.extend(ac.Stim_Frame_Align['Run008'][c_stim])
color_subgraph = Graph_Subtraction(raw_color_run,red_ids,white_ids,n_std)
#%% Plot 3 sub graphs in type of Z graph.

plt.clf()
plt.cla()
value_max = 800
value_min = -800

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9,4))
cbar_ax = fig.add_axes([.91, .15, .015, .7])
sns.heatmap(od_subgraph,center = 0,ax = axes[0],xticklabels=False,yticklabels=False,square=True,vmax = value_max,vmin = value_min,cbar_ax= cbar_ax)
sns.heatmap(ao_subgraph,center = 0,ax = axes[1],xticklabels=False,yticklabels=False,square=True,vmax = value_max,vmin = value_min,cbar_ax= cbar_ax)
sns.heatmap(color_subgraph,center = 0,ax = axes[2],xticklabels=False,yticklabels=False,square=True,vmax = value_max,vmin = value_min,cbar_ax= cbar_ax)

axes[0].set_title('LE - RE',size = 14)
axes[1].set_title('Orien 45 - 135',size = 14)
axes[2].set_title('Red - White',size = 14)
fig.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=0.88, top=None, hspace=None)
plt.show()
