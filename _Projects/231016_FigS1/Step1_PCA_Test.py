'''
This script tested basic PCA parameters, we have 

Do PCA on stim graph, get PC of specific meaning of stimulus on.

And train an SVM, classifiy cells on spon.

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
#%%  ################## Initialization #########################

work_path = r'D:\_Path_For_Figs\S1_PCA_Test'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Datas_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
cc_path = all_path_dic[2]
# Get all stim and spon frames in cell loc.
ac = ot.Load_Variable_v2(cc_path,'Cell_Class.pkl')
spon_frame = ot.Load_Variable(cc_path,'Spon_Before.pkl')
# get all spon graphs with ISI.
all_stim_frame,all_stim_label = ac.Combine_Frame_Labels(color = True)

#%%  ################## DO AND PLOT PCA #########################
#%% Do PCA on stim graph, and get visable graph.
from Advanced_Tools import Z_PCA
comps_stim,coords_stim,model = Z_PCA(Z_frame=all_stim_frame,sample='Frame')
# plot all PC graphs into a subfolder, and get all 
OD_map = ac.OD_t_graphs['OD'].loc['t_value']
HV_map = ac.Orien_t_graphs['H-V'].loc['t_value']
AO_map = ac.Orien_t_graphs['A-O'].loc['t_value']
Orien225_map = ac.Orien_t_graphs['Orien22.5-112.5'].loc['t_value']
Orien675_map = ac.Orien_t_graphs['Orien67.5-157.5'].loc['t_value']
Red_map = ac.Color_t_graphs['Red-White'].loc['t_value']
Green_map = ac.Color_t_graphs['Green-White'].loc['t_value']
Blue_map = ac.Color_t_graphs['Blue-White'].loc['t_value']

explained_var_ratio = model.explained_variance_ratio_
PC_similarity = pd.DataFrame(columns=['OD','HV','Orien22.5-112.5','A-O','Orien67.5-157.5','Red','Green','Blue'])
all_graph_list = [OD_map,HV_map,Orien225_map,AO_map,Orien675_map,Red_map,Green_map,Blue_map]
pca_path = work_path+'\\PCA_Components_Stim'
ot.mkdir(pca_path)
for i in tqdm(range(len(comps_stim))):
    c_pc = comps_stim[i,:]
    c_returned_graph = ac.Generate_Weighted_Cell(c_pc)
    plt.clf()
    plt.cla()
    fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (5,5),dpi = 180)
    sns.heatmap(c_returned_graph,center=0,square= True,xticklabels=False,yticklabels=False,ax = ax)
    ax.set_title(f'PC {i}')
    fig.figure.savefig(f'{pca_path}\\PC{i}.png')
    plt.close()
    # get map similarity matrix
    all_corrs = []
    for j,c_map in enumerate(all_graph_list):
        c_r,_ = pearsonr(np.array(c_map),c_pc)
        all_corrs.append(abs(c_r))
    PC_similarity.loc[len(PC_similarity),:] = all_corrs
ot.Save_Variable(work_path,'All_PC_Similarity',PC_similarity)
ot.Save_Variable(work_path,'PCA_Model',model)
ot.Save_Variable(work_path,'All_Stim_Coordinates',coords_stim[:,:20])
ot.Save_Variable(work_path,'PC_axis',comps_stim[:20,:])

#%% Plot PC components of all points into PC space.
def update(frame):
    ax.view_init(elev=25, azim=frame)  # Update the view angle for each frame
    return ax,
plt.clf()
plt.cla()
u = coords_stim
fig = plt.figure(figsize = (10,8))
ax = plt.axes(projection='3d')
ax.grid(False)
label = all_stim_label
# label = (all_stim_label<9)*(all_stim_label>0)
sc = ax.scatter3D(u[:,1], u[:,2], u[:,3],s = 5,c = label,cmap = 'jet')
ax.set_xlabel('PC 2')
ax.set_ylabel('PC 3')
ax.set_zlabel('PC 4')
animation = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=150)
animation.save(f'Plot_3D.gif', writer='pillow')
#%%################## PLOT STIM PC EMBEDDINGS ###################################
# Plot different graph on different color label.
#%% 1. Get OD colors.
import matplotlib.colors as mcolors
import colorsys
u = coords_stim
color_files = np.zeros(shape = (len(all_stim_label),3))
for i,c_label in enumerate(all_stim_label):
    if (c_label>0)*(c_label<9)*(c_label%2==1):# LE case
        color_files[i,:] = [1,0,0]# red
    elif (c_label>0)*(c_label<9)*(c_label%2==0):# RE
        color_files[i,:] = [0,1,0]# green
    else:
        color_files[i,:] = [0.5,0.5,0.5]
plt.clf()
plt.cla()  
fig = plt.figure(figsize = (8,5))
ax = plt.axes(projection='3d')
ax.grid(False)
sc = ax.scatter3D(u[:,1], u[:,2], u[:,3],s = 5,c = color_files)
custom_cmap = mcolors.ListedColormap([[1,0,0],[0,1,0],[0.5,0.5,0.5]])
cax = fig.add_axes([0.9, 0.3, 0.03, 0.4])
bounds = np.arange(0,4) # bound will have 1 more unit after graph.
norm = mpl.colors.BoundaryNorm(bounds, custom_cmap.N)
c_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=custom_cmap),cax=cax, label='Prefer Eye')
c_bar.set_ticks(np.arange(0,3)+0.5)
c_bar.set_ticklabels(['LE','RE','Non-Eye'])
c_bar.ax.tick_params(size=0)
ax.set_xlabel('PC 2')
ax.set_ylabel('PC 3')
ax.set_zlabel('PC 4')
ax.set_title('Eye prefered distribution-PCA')
fig.tight_layout()
plt.show()
animation = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=150)
animation.save(f'Plot_3D.gif', writer='pillow')
#%% 2. Get Orientation colors.
color_files_orien = np.zeros(shape = (len(all_stim_label),3))
for i,c_label in enumerate(all_stim_label):
    if (c_label>8)*(c_label<17):# Orientations 
        c_orien = 22.5*(c_label-9)
        c_hue = c_orien/180
        c_lightness = 0.5
        c_saturation = 1
        color_files_orien[i,:] = colorsys.hls_to_rgb(c_hue, c_lightness, c_saturation)
    else:
        color_files_orien[i,:] = colorsys.hls_to_rgb(0,0.5,0)
# plot orientations
plt.clf()
plt.cla()  
u = coords_stim
fig = plt.figure(figsize = (8,5))
ax = plt.axes(projection='3d')
ax.grid(False)
sc = ax.scatter3D(u[:,1], u[:,2], u[:,3],s = 5,c = color_files_orien)
color_sets = np.zeros(shape = (8,3))
for i,c_orien in enumerate(np.arange(0,180,22.5)):
    c_hue = c_orien/180
    c_lightness = 0.5
    c_saturation = 1
    color_sets[i,:] = colorsys.hls_to_rgb(c_hue, c_lightness, c_saturation)
custom_cmap = mcolors.ListedColormap(color_sets)
cax = fig.add_axes([0.9, 0.3, 0.03, 0.4])
bounds = np.arange(0,202.5,22.5) # bound will have 1 more unit after graph.
norm = mpl.colors.BoundaryNorm(bounds, custom_cmap.N)
c_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=custom_cmap),cax=cax, label='Prefer Orientation')
c_bar.set_ticks(np.arange(0,180,22.5)+11.25)
c_bar.set_ticklabels(np.arange(0,180,22.5))
c_bar.ax.tick_params(size=0)
ax.set_xlabel('PC 2')
ax.set_ylabel('PC 3')
ax.set_zlabel('PC 4')
ax.set_title('Orientation prefered distribution-PCA')
fig.tight_layout()
plt.show()
animation = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=150)
animation.save(f'Plot_3D.gif', writer='pillow')
#%%3.Get colors on PC5-7
color_files_hue = np.zeros(shape = (len(all_stim_label),3))
for i,c_label in enumerate(all_stim_label):
    if c_label==17:# Red
        color_files_hue[i,:] = [1,0,0]
    elif c_label == 18:
        color_files_hue[i,:] = [1,1,0]
    elif c_label == 19:
        color_files_hue[i,:] = [0,1,0]
    elif c_label == 20:
        color_files_hue[i,:] = [0,1,1]
    elif c_label == 21:
        color_files_hue[i,:] = [0,0,1]
    elif c_label == 22:
        color_files_hue[i,:] = [1,0,1]
    else:
        color_files_hue[i,:] = [0.5,0.5,0.5]
# plot colors
plt.clf()
plt.cla()  
u = coords_stim
fig = plt.figure(figsize = (8,5))
ax = plt.axes(projection='3d')
ax.grid(False)
sc = ax.scatter3D(u[:,1], u[:,2], u[:,3],s = 5,c = color_files_hue)
custom_cmap = mcolors.ListedColormap([[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1],[0.5,0.5,0.5]])
cax = fig.add_axes([0.9, 0.3, 0.03, 0.4])
bounds = np.arange(0,8) # bound will have 1 more unit after graph.
norm = mpl.colors.BoundaryNorm(bounds, custom_cmap.N)
c_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=custom_cmap),cax=cax, label='Prefer Color')
c_bar.set_ticks(np.arange(0,7)+0.5)
c_bar.set_ticklabels(np.arange(0,7))
c_bar.ax.tick_params(size=0)
ax.set_xlabel('PC 2')
ax.set_ylabel('PC 3')
ax.set_zlabel('PC 4')
ax.set_title('Color prefered distribution-PCA')
fig.tight_layout()
plt.show()
# animation = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=150)
# animation.save(f'Plot_3D.gif', writer='pillow')
#%%################# SPON EMBEDDING ########################################
# After stim description,we do spon here. embedding spon onto the stim PC space, and use SVM to classification.
#  train an svc and do reduction.
spon_embeddings = model.transform(spon_frame)
used_dims = coords_stim[:,1:20]
classifier,score = SVM_Classifier(used_dims,all_stim_label)
predicted_spon_frame = SVC_Fit(classifier=classifier,data = spon_embeddings[:,1:20],thres_prob=0)
# get all spon embedded frames.
plt.clf()
plt.cla()
u = spon_embeddings
fig = plt.figure(figsize = (10,8))
ax = plt.axes(projection='3d')
ax.grid(False)
sc = ax.scatter3D(u[:,1], u[:,2], u[:,3],s = 5,c = predicted_spon_frame,cmap = 'rainbow')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
animation = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=150)
animation.save(f'Plot_3D.gif', writer='pillow')

#%%######################### VECTOR ANAYISIS ##################################
u = coords_stim[:,1:20] # vector of all stim infos.
LE_locs = np.where((all_stim_label>0)*(all_stim_label<9)*(all_stim_label%2 == 1))[0] # LE Stim Frames
LE_vec = u[LE_locs,:].mean(0)
LE_vec = LE_vec/np.linalg.norm(LE_vec)
RE_locs = np.where((all_stim_label>0)*(all_stim_label<9)*(all_stim_label%2 == 0))[0] # RE Stim Frames
RE_vec = u[RE_locs,:].mean(0)
RE_vec = RE_vec/np.linalg.norm(RE_vec)
LR_angle = np.arccos(np.dot(LE_vec,RE_vec))*180/np.pi
print(f'LE and RE have Angle {LR_angle:.2f}')

