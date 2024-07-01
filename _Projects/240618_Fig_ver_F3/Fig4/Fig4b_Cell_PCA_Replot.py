

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
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import colorsys
import matplotlib as mpl

work_path = r'D:\_Path_For_Figs\240520_Figs_ver_F1\Fig4_Cell_In_Spon'
all_path_dic = list(ot.Get_Subfolders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)

#%% Useful Functions
def Angle_Calculate(vec1,vec2): # this function will calculate angle between 2 given vecs.
    norm_v1 = vec1/np.linalg.norm(vec1)
    norm_v2 = vec2/np.linalg.norm(vec2)
    cos = np.dot(norm_v1,norm_v2)
    angle = np.arccos(cos)*180/np.pi
    return angle,cos

def Vector_AVR_Response(vec,pc_comps,spon_frame,ac,centerlize = True): # this will generate average response 
    current_vec_frames = np.dot(pc_comps.T,vec)
    avr_response = np.dot(spon_frame.T,current_vec_frames)
    if centerlize == True:
        avr_response = avr_response-avr_response.mean()
    avr_graph = ac.Generate_Weighted_Cell(avr_response)
    return avr_response,avr_graph

def Vector_Caculator(all_pc_coords,posi_cell_lists,nega_cell_lists = []):
    all_pc_coords = np.array(all_pc_coords)
    posi_coords = np.nan_to_num(all_pc_coords[posi_cell_lists,:].mean(0))
    nega_coords = np.nan_to_num(all_pc_coords[nega_cell_lists,:].mean(0))
    general_corr = posi_coords-nega_coords
    return general_corr

#%% 
'''
Fig 4B, Use Location 18M as example, we Generate Cellular PCA Explained VAR.
'''

used_pc_num = 20
expt_folder = all_path_dic[2]
c_spon = np.array(ot.Load_Variable_v2(expt_folder,'Spon_Before.pkl'))
c_ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
c_pc_comps,c_pc_coords,c_pc_model = Z_PCA(c_spon,sample='Cell',pcnum=used_pc_num)
print(f'Total {used_pc_num} explained VAR {c_pc_model.explained_variance_ratio_.sum()*100:.1f}%')
# get tuing index of all cells.
ac_oriens = c_ac.all_cell_tunings.loc['Best_Orien',:]
ac_ods = c_ac.all_cell_tunings.loc['OD',:]

# get vecs of OD,HV,AO.
LE_cells = np.where(c_ac.all_cell_tunings.loc['Best_Eye',:]=='LE')[0] # this is absolute id.
RE_cells = np.where(c_ac.all_cell_tunings.loc['Best_Eye',:]=='RE')[0] # this is absolute id.
OD_vec = Vector_Caculator(c_pc_coords,LE_cells,RE_cells)
Orien0_cells = np.where(ac_oriens == 'Orien0')[0]
Orien90_cells = np.where(ac_oriens == 'Orien90')[0]
HV_vec = Vector_Caculator(c_pc_coords,Orien0_cells,Orien90_cells)
Orien45_cells = np.where(ac_oriens == 'Orien45')[0]
Orien135_cells = np.where(ac_oriens == 'Orien135')[0]
AO_vec = Vector_Caculator(c_pc_coords,Orien45_cells,Orien135_cells)

#%%
'''
Fig 4CA, We scatter OD Graphs.
'''
OD_PCs = [0,1,2]

import matplotlib.cm as cm
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
import matplotlib as mpl
import colorsys

plt.clf()
plt.cla()
zoom = 1
od_elev = 15
od_azim = 75
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (6,6),dpi = 300,subplot_kw=dict(projection='3d'))

# Grid Preparing
ax.view_init(elev=od_elev, azim=od_azim)
plotted_pcs = OD_PCs
# ax.set_xlabel(f'PC {plotted_pcs[0]+1}',size = 12)
# ax.set_ylabel(f'PC {plotted_pcs[1]+1}',size = 12)
# ax.set_zlabel(f'PC {plotted_pcs[2]+1}',size = 12)

ax.grid(False)
ax.set_box_aspect(aspect=None, zoom=zoom) # shrink graphs
# ax[i].axes.set_xlim3d(left=-15, right=30)
# ax[i].axes.set_ylim3d(bottom=-25, top=25)
# ax[i].axes.set_zlim3d(bottom=-20, top=20)
# ax[i].set_position([ax[i].get_position().x0-0.12, ax[i].get_position().y0, ax[i].get_position().width*zoom, ax[i].get_position().height*zoom])
# set z label location
tmp_planes = ax.zaxis._PLANES 
ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3], 
                    tmp_planes[0], tmp_planes[1], 
                    tmp_planes[4], tmp_planes[5])
sc = ax.scatter3D(c_pc_coords[:,OD_PCs[0]], c_pc_coords[:,OD_PCs[1]], c_pc_coords[:,OD_PCs[2]],s = 20,lw=0,c = np.array(ac_ods),cmap = 'bwr',vmin = -2,vmax = 1.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
fig.tight_layout()

#%% Plot color bars
value_max = 1.5
value_min = -2
plt.clf()
plt.cla()
data = [[value_min, value_max], [value_min, value_max]]
# Create a heatmap
fig, ax = plt.subplots(figsize = (2,1),dpi = 300)
# fig2, ax2 = plt.subplots()
g = sns.heatmap(data, center=0,ax = ax,vmax = value_max,vmin = value_min,cbar_kws={"aspect": 5,"shrink": 1,"orientation": "vertical"},cmap = 'bwr')
# Hide the heatmap itself by setting the visibility of its axes
ax.set_visible(False)
g.collections[0].colorbar.set_ticks([value_min,0,value_max])
g.collections[0].colorbar.set_ticklabels([value_min,0,value_max])
g.collections[0].colorbar.ax.tick_params(labelsize=8)
g.collections[0].colorbar.set_label(label='',size=9)
plt.show()

#%%
'''
Fig 4CB, We scatter Orientation Graphs.
'''

Orien_PCs = [1,2,3]
plt.clf()
plt.cla()
orien_elev = 15
orien_azim = 240
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (6,6),dpi = 300,subplot_kw=dict(projection='3d'))

# Grid Preparing
ax.view_init(elev=orien_elev, azim=orien_azim)
plotted_pcs = Orien_PCs
# ax.set_xlabel(f'PC {plotted_pcs[0]+1}',size = 12)
# ax.set_ylabel(f'PC {plotted_pcs[1]+1}',size = 12)
# ax.set_zlabel(f'PC {plotted_pcs[2]+1}',size = 12)
ax.grid(False)
ax.set_box_aspect(aspect=None, zoom=1) # shrink graphs
# ax[i].axes.set_xlim3d(left=-15, right=30)
# ax[i].axes.set_ylim3d(bottom=-25, top=25)
# ax[i].axes.set_zlim3d(bottom=-20, top=20)
# ax[i].set_position([ax[i].get_position().x0-0.12, ax[i].get_position().y0, ax[i].get_position().width*zoom, ax[i].get_position().height*zoom])
# set z label location
tmp_planes = ax.zaxis._PLANES 
ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3], 
                    tmp_planes[0], tmp_planes[1], 
                    tmp_planes[4], tmp_planes[5])

## get orien color bars.
color_setb = np.zeros(shape = (8,3))
for i,c_orien in enumerate(np.arange(0,180,22.5)):
    c_hue = c_orien/180
    c_lightness = 0.5
    c_saturation = 1
    color_setb[i,:] = colorsys.hls_to_rgb(c_hue, c_lightness, c_saturation)

orien_color = np.zeros(shape=(len(ac_oriens),3), dtype='f8')
for i in range(len(ac_oriens)):
    c_orien = ac_oriens.iloc[i]
    if c_orien == 'False':
        orien_color[i,:] = [0.7,0.7,0.7]
    else:
        orien_color[i,:] = color_setb[int(float(ac_oriens.iloc[i][5:])/22.5),:]
ax.scatter3D(c_pc_coords[:,Orien_PCs[0]],c_pc_coords[:,Orien_PCs[1]],c_pc_coords[:,Orien_PCs[2]],s = 20,lw=0,c = orien_color)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
fig.tight_layout()

