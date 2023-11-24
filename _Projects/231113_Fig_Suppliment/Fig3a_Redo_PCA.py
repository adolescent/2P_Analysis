'''
This script will redo the Fig 3A Result, not using UMAP, but PCA instead.

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
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Datas_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
#%% Useful functions.
import colorsys
def Tuning_Color_Bar(ac):
    cell_tuning_colorbar = pd.DataFrame(index= ac.acn,columns=['Dist','OD','Orien','Orien_Hue_R','Orien_Hue_G','Orien_Hue_B'])
    cc_tunings = ac.all_cell_tunings
    cell_tuning_colorbar['OD'] = cc_tunings.loc['OD',:]
    for i,cc in enumerate(ac.acn):
        cc_loc = ac.Cell_Locs[cc]
        c_dist = np.sqrt(cc_loc['X']*cc_loc['X']+cc_loc['Y']*cc_loc['Y'])
        cell_tuning_colorbar.loc[cc,'Dist'] = c_dist

    c_Best_Orien = cc_tunings.loc['Best_Orien',:]
    cell_tuning_colorbar['Orien'] = c_Best_Orien
    for i,cc in enumerate(ac.acn):
        c_orien = c_Best_Orien.loc[cc]
        if c_orien == 'False':
            c_orien_color = colorsys.hls_to_rgb(0,0.5,0)
        else:
            c_orien = float(c_orien[5:])
            c_hue = c_orien/180
            c_lightness = 0.5
            c_saturation = 1
            c_orien_color = colorsys.hls_to_rgb(c_hue, c_lightness, c_saturation)
        cell_tuning_colorbar.loc[cc,['Orien_Hue_R','Orien_Hue_G','Orien_Hue_B']] = c_orien_color

    return cell_tuning_colorbar

def OD_Graph_Plotter(scatter,od_bar,save_path,save_gif = True,pca_range = [1,2,3]):
    plt.clf()
    plt.cla()
    fig = plt.figure(figsize = (10,8))
    ax = plt.axes(projection='3d')
    ax.grid(False)
    sc = ax.scatter3D(scatter[:,0], scatter[:,1], scatter[:,2],s = 5,c = od_bar,cmap = 'bwr')
    cbar = fig.colorbar(sc, shrink=0.5)
    cbar.set_label('OD Tuning')
    ax.set_xlabel(f'PC {pca_range[0]}')
    ax.set_ylabel(f'PC {pca_range[1]}')
    ax.set_zlabel(f'PC {pca_range[2]}')
    ax.set_title('OD Labeled Neuron in PC Space')
    azim_list = [30,60,120,150,210,240,300,330]
    for i,c_azim in enumerate(azim_list):
        ax.view_init(elev=25, azim=c_azim)
        plt.savefig(f'{save_path}\\OD_Plot_{c_azim}.png', dpi=180)
    if save_gif == True:
        def update(frame):
            ax.view_init(elev=25, azim=frame)  # Update the view angle for each frame
            return ax,
        animation = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=150)
        animation.save(f'{save_path}\\OD_Plot_3D.gif', writer='pillow')
        
def Orien_Graph_Plotter(scatter,orien_bar,save_path,save_gif = True,pca_range = [2,3,4]):# work only in this script
    plt.clf()
    plt.cla()
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    sc = ax.scatter3D(scatter[:,0], scatter[:,1], scatter[:,2],s = 5,c = orien_bar)
    # Add a legend. This is a little fuzzy.
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
    ax.set_xlabel(f'PC {pca_range[0]}')
    ax.set_ylabel(f'PC {pca_range[1]}')
    ax.set_zlabel(f'PC {pca_range[2]}')
    ax.set_title('Orientation Labeled Neuron in PC Space')
    azim_list = [30,60,120,150,210,240,300,330]
    for i,c_azim in enumerate(azim_list):
        ax.view_init(elev=25, azim=c_azim)
        plt.savefig(f'{save_path}\\Orientation_Plot_{c_azim}.png', dpi=180)
    # and a gif.
    if save_gif == True:
        def update(frame):
            ax.view_init(elev=25, azim=frame)  # Update the view angle for each frame
            return ax,
        animation = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=150)
        animation.save(f'{save_path}\\Orientation_Plot_3D.gif', writer='pillow')




#%%############################## STEP1 PCA STIM SHOW #############################
'''
This part will show PCA seperation on stim data, this will be very clear, as tuning preference easy to find.
'''
from Advanced_Tools import Z_PCA
used_pc_num = 20
all_pca_result_stim = {}
all_pca_result_spon = {}
for i,cloc in tqdm(enumerate(all_path_dic)):
    # get folder 
    cname = cloc.split('\\')[-1]
    c_cond_path = ot.join(work_path,cname)
    ot.mkdir(c_cond_path)
    c_stim_path_od = ot.join(c_cond_path,'OD_Embedding_Stim')
    ot.mkdir(c_stim_path_od)
    # read in variable
    c_spon_frame = ot.Load_Variable(cloc,'Spon_Before.pkl')
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_tuning_colorbar = Tuning_Color_Bar(ac) # this is tuning info of all graph.
    c_stim_frame,c_stim_label = ac.Combine_Frame_Labels(color = False)
    # Plot OD Embedding Stim
    comps_stim,coords_stim,model = Z_PCA(Z_frame=c_stim_frame,sample='Cell')
    all_pca_result_stim[cname]={}
    all_pca_result_stim[cname]['All_PCs']=comps_stim[:used_pc_num,:]
    all_pca_result_stim[cname]['All_embeddings']=coords_stim[:,:used_pc_num]
    all_pca_result_stim[cname]['PCA_Model']=model
    # c_scatter = coords_stim[:,:3]
    OD_Graph_Plotter(scatter=coords_stim[:,0:3],od_bar=np.array(c_tuning_colorbar['OD']),save_path=c_stim_path_od,save_gif = True,pca_range=[1,2,3])
    # And Orien Embedding Stim
    c_stim_path_orien = ot.join(c_cond_path,'Orien_Embedding_Stim')
    ot.mkdir(c_stim_path_orien)
    Orien_Graph_Plotter(scatter=coords_stim[:,1:4],orien_bar=np.array(c_tuning_colorbar[['Orien_Hue_R','Orien_Hue_G','Orien_Hue_B']]),save_path=c_stim_path_orien,save_gif = True,pca_range=[2,3,4])
    ##### Then Plot OD Embedding Spon
    comps_spon,coords_spon,model_spon = Z_PCA(Z_frame=c_spon_frame,sample='Cell')
    all_pca_result_spon[cname]={}
    all_pca_result_spon[cname]['All_PCs']=comps_spon[:used_pc_num,:]
    all_pca_result_spon[cname]['All_embeddings']=coords_spon[:,:used_pc_num]
    all_pca_result_spon[cname]['PCA_Model']=model_spon
    c_spon_path_od = ot.join(c_cond_path,'OD_Embedding_Spon')
    c_spon_path_orien = ot.join(c_cond_path,'Orien_Embedding_Spon')
    ot.mkdir(c_spon_path_od)
    ot.mkdir(c_spon_path_orien)
    # c_scatter = coords_spon[:,:3]
    OD_Graph_Plotter(scatter=coords_spon[:,:3],od_bar=np.array(c_tuning_colorbar['OD']),save_path=c_spon_path_od,save_gif = True,pca_range=[1,2,3])
    Orien_Graph_Plotter(scatter=coords_spon[:,3:6],orien_bar=np.array(c_tuning_colorbar[['Orien_Hue_R','Orien_Hue_G','Orien_Hue_B']]),save_path=c_spon_path_orien,save_gif = True,pca_range=[4,5,6])
    # And Orien Embedding Spon

ot.Save_Variable(work_path,'All_PCA_Result_Stim',all_pca_result_stim)
ot.Save_Variable(work_path,'All_PCA_Result_Spon',all_pca_result_spon)

#%%
cc_tunings = ac.all_cell_tunings
c_OD_index = np.array(cc_tunings.loc['OD',:])
plt.clf()
plt.cla()
comps_stim,coords_stim,model = Z_PCA(Z_frame=c_spon_frame,sample='Cell')
u = coords_stim[:,:3]
fig = plt.figure(figsize = (10,8))
ax = plt.axes(projection='3d')
ax.grid(False)
sc = ax.scatter3D(u[:,0], u[:,1], u[:,2],s = 5,c = c_OD_index,cmap = 'bwr')
cbar = fig.colorbar(sc, shrink=0.5)
cbar.set_label('OD t value')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')
ax.set_title('Eye-Preference Labeled Neuron Embeddings')

