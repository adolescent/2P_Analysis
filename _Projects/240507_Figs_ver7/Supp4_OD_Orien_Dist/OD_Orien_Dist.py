'''

This script will do PCA on cell space, and get OD-HV-AO dist and their distance.

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
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import colorsys
import matplotlib as mpl

work_path = r'D:\_Path_For_Figs\230507_Figs_v7\Cell_PCA_oriens'
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
    if nega_cell_lists != []:
        nega_coords = np.nan_to_num(all_pc_coords[nega_cell_lists,:].mean(0))
        general_corr = posi_coords-nega_coords
    else:
        general_corr = posi_coords
    return general_corr

#%% 
'''
Step1, generate example embedding of cells, and get different cell coords.
'''

used_pc_num = 20
expt_folder = all_path_dic[2]
c_spon = np.array(ot.Load_Variable_v2(expt_folder,'Spon_Before.pkl'))
c_ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
c_pc_comps,c_pc_coords,c_pc_model = Z_PCA(c_spon,sample='Cell',pcnum=used_pc_num)

ac_oriens = c_ac.all_cell_tunings.loc['Best_Orien',:]
ac_ods = c_ac.all_cell_tunings.loc['OD',:]
#%%%
# get vecs of OD,HV,AO.
LE_cells = np.where(c_ac.all_cell_tunings.loc['Best_Eye',:]=='LE')[0] # this is absolute id.
RE_cells = np.where(c_ac.all_cell_tunings.loc['Best_Eye',:]=='RE')[0] # this is absolute id.
LE_vec = Vector_Caculator(c_pc_coords,LE_cells)
RE_vec = Vector_Caculator(c_pc_coords,RE_cells)
Orien0_cells = np.where(ac_oriens == 'Orien0')[0]
Orien90_cells = np.where(ac_oriens == 'Orien90')[0]
Orien45_cells = np.where(ac_oriens == 'Orien45')[0]
Orien135_cells = np.where(ac_oriens == 'Orien135')[0]

Orien0_vec = Vector_Caculator(c_pc_coords,Orien0_cells)
Orien45_vec = Vector_Caculator(c_pc_coords,Orien45_cells)
Orien90_vec = Vector_Caculator(c_pc_coords,Orien90_cells)
Orien135_vec = Vector_Caculator(c_pc_coords,Orien135_cells)
#%% ######################### Fig 4a EXAMPLE LOCATION  ############################

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

#%% Get Distance of H-V,A-O,LE-RE.
dist_od = np.linalg.norm(LE_vec-RE_vec)
dist_hv = np.linalg.norm(Orien0_vec-Orien90_vec)
dist_ao = np.linalg.norm(Orien45_vec-Orien135_vec)

#%%
'''
Step 2, do stats of all data point, and get OD-Orientation Distance relationship.
'''

all_explained_var = []
all_dist = pd.DataFrame(columns = ['Loc','LE-RE','Orien0-90','Orien45-135'])
used_pc_num = 20

for i,cloc in tqdm(enumerate(all_path_dic)):

    cloc_name = cloc.split('\\')[-1]
    c_spon = np.array(ot.Load_Variable_v2(cloc ,'Spon_Before.pkl'))
    c_ac = ot.Load_Variable_v2(cloc ,'Cell_Class.pkl')
    c_pc_comps,c_pc_coords,c_pc_model = Z_PCA(c_spon,sample='Cell',pcnum=used_pc_num)
    ac_oriens = c_ac.all_cell_tunings.loc['Best_Orien',:]
    ac_ods = c_ac.all_cell_tunings.loc['OD',:]
    all_explained_var.append(c_pc_model.explained_variance_ratio_.sum())

    LE_cells = np.where(c_ac.all_cell_tunings.loc['Best_Eye',:]=='LE')[0] # this is absolute id.
    RE_cells = np.where(c_ac.all_cell_tunings.loc['Best_Eye',:]=='RE')[0] # this is absolute id.
    LE_vec = Vector_Caculator(c_pc_coords,LE_cells)
    RE_vec = Vector_Caculator(c_pc_coords,RE_cells)
    Orien0_cells = np.where(ac_oriens == 'Orien0')[0]
    Orien90_cells = np.where(ac_oriens == 'Orien90')[0]
    Orien45_cells = np.where(ac_oriens == 'Orien45')[0]
    Orien135_cells = np.where(ac_oriens == 'Orien135')[0]

    Orien0_vec = Vector_Caculator(c_pc_coords,Orien0_cells)
    Orien45_vec = Vector_Caculator(c_pc_coords,Orien45_cells)
    Orien90_vec = Vector_Caculator(c_pc_coords,Orien90_cells)
    Orien135_vec = Vector_Caculator(c_pc_coords,Orien135_cells)

    dist_od = np.linalg.norm(LE_vec-RE_vec)
    dist_hv = np.linalg.norm(Orien0_vec-Orien90_vec)
    dist_ao = np.linalg.norm(Orien45_vec-Orien135_vec)

    all_dist.loc[len(all_dist),:] = [cloc_name,dist_od,dist_hv,dist_ao]

#%% Plot part, plot dist between od and orien.
all_dist['HV_Ratio'] = all_dist['Orien0-90']/all_dist['LE-RE']
all_dist['AO_Ratio'] = all_dist['Orien45-135']/all_dist['LE-RE']
plotable = pd.melt(all_dist,id_vars=['Loc'], value_vars=['HV_Ratio', 'AO_Ratio'],value_name='Ratio',var_name='Network')

plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (3,5),dpi = 180)
ax.axhline(y = 1,color = 'gray',linestyle = '--')
sns.barplot(data = plotable, x= 'Network',y = 'Ratio',ax = ax,width=0.5,hue = 'Network')
