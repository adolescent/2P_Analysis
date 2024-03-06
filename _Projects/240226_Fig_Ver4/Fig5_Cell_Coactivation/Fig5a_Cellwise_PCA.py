'''
This script do PCA on cellular, try to get the activation similarity between cells.

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
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import colorsys
import matplotlib as mpl

work_path = r'D:\_Path_For_Figs\Fig3_Redo'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
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

#%% ######################  Example Location PCA #################################
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

#%% ########################### Fig 5a EXAMPLE LOCATION  ##################################

OD_PCs = [0,1,2]
Orien_PCs = [1,2,3]

import matplotlib.cm as cm
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
import matplotlib as mpl
import colorsys
# define several functions here.

plt.clf()
plt.cla()
zoom = 1
od_elev = 15
od_azim = 75

orien_elev = 15
orien_azim = 240

fig,ax = plt.subplots(nrows=1, ncols=2,figsize = (10,5),dpi = 180,subplot_kw=dict(projection='3d'))

# Grid Preparing
ax[0].view_init(elev=od_elev, azim=od_azim)
ax[1].view_init(elev=orien_elev, azim=orien_azim)

for i in range(2):
    plotted_pcs = [OD_PCs,Orien_PCs][i]
    ax[i].set_xlabel(f'PC {plotted_pcs[0]+1}')
    ax[i].set_ylabel(f'PC {plotted_pcs[1]+1}')
    ax[i].set_zlabel(f'PC {plotted_pcs[2]+1}')
    ax[i].grid(False)
    ax[i].set_box_aspect(aspect=None, zoom=1) # shrink graphs
    # ax[i].axes.set_xlim3d(left=-15, right=30)
    # ax[i].axes.set_ylim3d(bottom=-25, top=25)
    # ax[i].axes.set_zlim3d(bottom=-20, top=20)
    # ax[i].set_position([ax[i].get_position().x0-0.12, ax[i].get_position().y0, ax[i].get_position().width*zoom, ax[i].get_position().height*zoom])
    # set z label location
    tmp_planes = ax[i].zaxis._PLANES 
    ax[i].zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                            tmp_planes[0], tmp_planes[1], 
                            tmp_planes[4], tmp_planes[5])



# plot OD graphs, use contineous color bars.
cbarax = fig.add_axes([0.04, 0.4, 0.015, 0.3])
sc = ax[0].scatter3D(c_pc_coords[:,OD_PCs[0]], c_pc_coords[:,OD_PCs[1]], c_pc_coords[:,OD_PCs[2]],s = 1,c = np.array(ac_ods),cmap = 'bwr')
cbar = fig.colorbar(sc, shrink=0.3,cax = cbarax)
cbar.set_label('OD Tuning')


## get orien color bars.
color_setb = np.zeros(shape = (8,3))
for i,c_orien in enumerate(np.arange(0,180,22.5)):
    c_hue = c_orien/180
    c_lightness = 0.5
    c_saturation = 1
    color_setb[i,:] = colorsys.hls_to_rgb(c_hue, c_lightness, c_saturation)
cax_b = fig.add_axes([0.93, 0.4, 0.015, 0.3])
custom_cmap = mcolors.ListedColormap(color_setb)
bounds = np.arange(0,202.5,22.5)
norm = mpl.colors.BoundaryNorm(bounds, custom_cmap.N)
c_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=custom_cmap),cax=cax_b, label='Orientation')
c_bar.set_ticks(np.arange(0,180,22.5)+11.25)
c_bar.set_ticklabels(np.arange(0,180,22.5))
c_bar.ax.tick_params(size=0)

# get color mat of all cell.
orien_color = np.zeros(shape=(len(ac_oriens),3), dtype='f8')
for i in range(len(ac_oriens)):
    c_orien = ac_oriens.iloc[i]
    if c_orien == 'False':
        orien_color[i,:] = [0.7,0.7,0.7]
    else:
        orien_color[i,:] = color_setb[int(float(ac_oriens.iloc[i][5:])/22.5),:]
ax[1].scatter3D(c_pc_coords[:,Orien_PCs[0]],c_pc_coords[:,Orien_PCs[1]],c_pc_coords[:,Orien_PCs[2]],s = 1,c = orien_color)

fig.suptitle('Cells Spontaneous Response In PCA Space',size = 16,y = 0.85)
ax[0].set_title('Distribution with Eye Preference',size = 10,y = 0.9)
ax[1].set_title('Distribution with Orientation Preference',size = 10,y = 0.9)


#%% ############################# FIG S5A - 3 Axis Recovered Response##########################
OD_vec_response,OD_vec_map =  Vector_AVR_Response(vec= OD_vec,pc_comps=c_pc_comps,spon_frame=c_spon,ac = c_ac)
HV_vec_response,HV_vec_map =  Vector_AVR_Response(vec= HV_vec,pc_comps=c_pc_comps,spon_frame=c_spon,ac = c_ac)
AO_vec_response,AO_vec_map =  Vector_AVR_Response(vec= AO_vec,pc_comps=c_pc_comps,spon_frame=c_spon,ac = c_ac)
# real stim funcmaps.
OD_stimmap = c_ac.Generate_Weighted_Cell(c_ac.OD_t_graphs['OD'].loc['CohenD'])
HV_stimmap = c_ac.Generate_Weighted_Cell(c_ac.Orien_t_graphs['H-V'].loc['CohenD'])
AO_stimmap = c_ac.Generate_Weighted_Cell(c_ac.Orien_t_graphs['A-O'].loc['CohenD'])
#%% Plot Stim maps with PC axes
graph_lists = ['OD','HV','AO']
plt.clf()
plt.cla()
value_max = 1.3
value_min = -1
font_size = 12
fig,axes = plt.subplots(nrows=2, ncols=3,figsize = (10,6),dpi = 180)
cbar_ax = fig.add_axes([.97, .15, .02, .7])
all_pc_axes = [OD_vec_map,HV_vec_map,AO_vec_map]
all_stim_graphs = [OD_stimmap,HV_stimmap,AO_stimmap]

for i,c_map in enumerate(graph_lists):
    c_pc_map = all_pc_axes[i]/all_pc_axes[i].max()
    sns.heatmap(c_pc_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    c_stim_map = all_stim_graphs[i]/all_stim_graphs[i].max()
    sns.heatmap(c_stim_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    axes[0,i].set_title(c_map,size = font_size)

axes[0,0].set_ylabel('PCA Functional Axes',rotation=90,size = font_size)
axes[1,0].set_ylabel('Stimulus Response',rotation=90,size = font_size)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
fig.tight_layout()
plt.show()

#%% ############################# Fig 5c ALL Loc Network_Corr_Angle ########################
all_angles = pd.DataFrame(columns = ['Axes','Angle','Corr'])
used_pc_num = 20
all_explained_var = np.zeros(len(all_path_dic))
for i,cloc in enumerate(all_path_dic):
    c_spon = np.array(ot.Load_Variable_v2(cloc,'Spon_Before.pkl'))
    c_ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_pc_comps,c_pc_coords,c_pc_model = Z_PCA(c_spon,sample='Cell',pcnum=used_pc_num)
    print(f'Total {used_pc_num} explained VAR {c_pc_model.explained_variance_ratio_.sum()*100:.1f}%')
    all_explained_var[i] = c_pc_model.explained_variance_ratio_.sum()
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

    all_angles.loc[len(all_angles),:] = ['HV-AO',Angle_Calculate(HV_vec,AO_vec)[0],Angle_Calculate(HV_vec,AO_vec)[1]]
    all_angles.loc[len(all_angles),:] = ['OD-AO',Angle_Calculate(OD_vec,AO_vec)[0],Angle_Calculate(OD_vec,AO_vec)[1]]
    all_angles.loc[len(all_angles),:] = ['OD-HV',Angle_Calculate(OD_vec,HV_vec)[0],Angle_Calculate(OD_vec,HV_vec)[1]]

#%% Plot angles.
plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (2,5),dpi = 180)
ax.axhline(y = 90,color='gray', linestyle='--')
sns.boxplot(data = all_angles,y = 'Angle',x = 'Axes',ax = ax,showfliers = 0,legend = True,width=0.5,palette="tab10")

ax.set_ylim(40,130)
ax.set_title('Functional Axes Included Angle',size = 12,y = 1.05)
ax.set_xlabel('Axes Pair',weight = 'bold')
ax.set_ylabel('Angle',weight = 'bold')
plt.show()