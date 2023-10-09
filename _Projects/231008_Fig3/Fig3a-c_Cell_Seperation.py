'''
This script will use UMAP to seperate cells in all spon series, we expect to get OD Orien seperated cell in all data points.
This result will give us the evidence of tuning's importance on spontaneous activity.
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

work_path = r'D:\_Path_For_Figs\Fig3_Cell_Seperation'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Datas_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)

#%% Cell Seperator. Reduce all cell to 2D using UMAP, and plot color with Eye, Orien, Distance 
import colorsys
cell_seperation_dic = {}
for i,c_loc in tqdm(enumerate(all_path_dic)):
    c_loc_name = c_loc.split('\\')[-1]
    cell_seperation_dic[c_loc_name] = {}
    c_spon_frame = ot.Load_Variable(c_loc,'Spon_Before.pkl')
    c_ac = ot.Load_Variable_v2(c_loc,'Cell_Class.pkl')
    cc_tunings = c_ac.all_cell_tunings
    # dist correlation
    dist = np.zeros(c_ac.cellnum)
    for i,cc in enumerate(c_ac.acn):
        c_loc = c_ac.Cell_Locs[cc]
        c_dist = np.sqrt(c_loc['X']^2+c_loc['Y']^2)
        dist[i] = c_dist
    del c_ac # save memory
    # get cell seperator and all cell embeddings.
    reducer_cell = umap.UMAP(n_components=3,n_neighbors=40)
    reducer_cell.fit(c_spon_frame.T)
    u = reducer_cell.embedding_
    c_OD_index = cc_tunings.loc['OD',:]
    c_Best_Orien = cc_tunings.loc['Best_Orien',:]
    c_orien_color = np.zeros(shape = (len(c_Best_Orien),3))
    for i in range(len(c_Best_Orien)):
        c_orien = c_Best_Orien.iloc[i]
        if c_orien == 'False':
            c_orien_color[i,:] = colorsys.hls_to_rgb(0,0.5,0)
        else:
            c_orien = float(c_orien[5:])
            c_hue = c_orien/180
            c_lightness = 0.5
            c_saturation = 1
            c_orien_color[i,:] = colorsys.hls_to_rgb(c_hue, c_lightness, c_saturation)
    # save labels into dict.
    cell_seperation_dic[c_loc_name]['reducer']= reducer_cell
    cell_seperation_dic[c_loc_name]['embeddings']= u
    cell_seperation_dic[c_loc_name]['Dist'] = dist
    cell_seperation_dic[c_loc_name]['OD_t'] = c_OD_index
    cell_seperation_dic[c_loc_name]['Orien'] = c_Best_Orien
    cell_seperation_dic[c_loc_name]['Orien_Hue'] = c_orien_color
ot.Save_Variable(work_path,'Cell_Seperate_Colors',cell_seperation_dic)

    # adjust orientation into
    # fig,ax = plt.subplots(1,figsize = (6,6))
    # ax = plt.scatter(x = u[:,0],y = u[:,1],c = c_OD_index,cmap = 'bwr')
    # ax = plt.scatter(x = u[:,0],y = u[:,1],c = c_orien_color)
#%% data saved, now let's concentrate on how to visualize!
# plot a od-umap map
# plot a orien-umap map
# plot a dist-umap map
## all graph have a gif, 4 dir stable view.
### and make graphs above for all data points.
# c_loc = list(cell_seperation_dic.keys())[2]
for i,c_loc in tqdm(enumerate(cell_seperation_dic)):
    cc_path = work_path+'\\'+c_loc
    cc = cell_seperation_dic[c_loc]
    u = cc['embeddings'] # this is embedding of all cell into umap space.
    OD_bar = np.array(cc['OD_t'])
    dist_bar= np.array(cc['Dist'])
    orien_bar = cc['Orien_Hue']
    orien_angle = cc['Orien']
    # GIF Updater
    def update(frame):
        ax.view_init(elev=25, azim=frame)  # Update the view angle for each frame
        return ax,
    # save OD graph
    ot.mkdir(cc_path)
    plt.clf()
    plt.cla()
    fig = plt.figure(figsize = (10,8))
    ax = plt.axes(projection='3d')
    ax.grid(False)
    sc = ax.scatter3D(u[:,0], u[:,1], u[:,2],s = 5,c = OD_bar,cmap = 'bwr')
    cbar = fig.colorbar(sc, shrink=0.5)
    cbar.set_label('OD t value')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')
    azim_list = [30,60,120,150,210,240,300,330]
    for i,c_azim in enumerate(azim_list):
        ax.view_init(elev=25, azim=c_azim)
        plt.savefig(f'{cc_path}\\OD_Plot_{c_azim}.png', dpi=180)
    # and a gif.
    animation = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=150)
    animation.save(f'{cc_path}\\OD_Plot_3D.gif', writer='pillow')
    ## Plot dist graph, all the same.
    plt.clf()
    plt.cla()
    fig = plt.figure(figsize = (10,8))
    ax = plt.axes(projection='3d')
    ax.grid(False)
    sc = ax.scatter3D(u[:,0], u[:,1], u[:,2],s = 5,c = dist_bar,cmap = 'bwr')
    cbar = fig.colorbar(sc, shrink=0.5)
    cbar.set_label('Distance')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')
    azim_list = [30,60,120,150,210,240,300,330]
    for i,c_azim in enumerate(azim_list):
        ax.view_init(elev=25, azim=c_azim)
        plt.savefig(f'{cc_path}\\Dist_Plot_{c_azim}.png', dpi=180)
    # and a gif.
    animation = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=150)
    animation.save(f'{cc_path}\\Dist_Plot_3D.gif', writer='pillow')
    ## Plot orientation here. The color bar need some change.
    plt.clf()
    plt.cla()
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    sc = ax.scatter3D(u[:,0], u[:,1], u[:,2],s = 5,c = orien_bar)
    # add manual color bar here.
    cax = fig.add_axes([0.9, 0.3, 0.03, 0.4])
    color_sets = np.zeros(shape = (8,3))
    for i,c_orien in enumerate(np.arange(0,180,22.5)):
        c_hue = c_orien/180
        c_lightness = 0.5
        c_saturation = 1
        color_sets[i,:] = colorsys.hls_to_rgb(c_hue, c_lightness, c_saturation)
    custom_cmap = mcolors.ListedColormap(color_sets)
    bounds = np.arange(0,202.5,22.5)
    norm = mpl.colors.BoundaryNorm(bounds, custom_cmap.N)
    c_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=custom_cmap),cax=cax, label='Prefer Orientation')
    c_bar.set_ticks(np.arange(0,180,22.5)+11.25)
    c_bar.set_ticklabels(np.arange(0,180,22.5))
    c_bar.ax.tick_params(size=0)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')
    azim_list = [30,60,120,150,210,240,300,330]
    for i,c_azim in enumerate(azim_list):
        ax.view_init(elev=25, azim=c_azim)
        plt.savefig(f'{cc_path}\\Orientation_Plot_{c_azim}.png', dpi=180)
    # and a gif.
    animation = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=150)
    animation.save(f'{cc_path}\\Orientation_Plot_3D.gif', writer='pillow')
    # set bar manually.
    
