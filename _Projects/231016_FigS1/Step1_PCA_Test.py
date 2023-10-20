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
sc = ax.scatter3D(u[:,0], u[:,1], u[:,2],s = 5,c = color_files)
custom_cmap = mcolors.ListedColormap([[1,0,0],[0,1,0],[0.5,0.5,0.5]])
cax = fig.add_axes([0.9, 0.3, 0.03, 0.4])
bounds = np.arange(0,4) # bound will have 1 more unit after graph.
norm = mpl.colors.BoundaryNorm(bounds, custom_cmap.N)
c_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=custom_cmap),cax=cax, label='Prefer Eye')
c_bar.set_ticks(np.arange(0,3)+0.5)
c_bar.set_ticklabels(['LE','RE','Non-Eye'])
c_bar.ax.tick_params(size=0)
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
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

#%%######################### VECTOR ANAYISIS ##################################
#%% 1.Vector Generation
# This part will generate vector of 
u = coords_stim[:,:20] # vector of all stim infos.
LE_locs = np.where((all_stim_label>0)*(all_stim_label<9)*(all_stim_label%2 == 1))[0] # LE Stim Frames
LE_vec = u[LE_locs,1:4].mean(0)
LE_vec = LE_vec/np.linalg.norm(LE_vec)
RE_locs = np.where((all_stim_label>0)*(all_stim_label<9)*(all_stim_label%2 == 0))[0] # RE Stim Frames
RE_vec = u[RE_locs,1:4].mean(0)
RE_vec = RE_vec/np.linalg.norm(RE_vec)
Orien0_locs = np.where((all_stim_label==9))[0]
Orien0_vec = u[Orien0_locs,1:4].mean(0)
Orien0_vec = Orien0_vec/np.linalg.norm(Orien0_vec)
Orien45_locs = np.where((all_stim_label==11))[0]
Orien45_vec = u[Orien45_locs,1:4].mean(0)
Orien45_vec = Orien45_vec/np.linalg.norm(Orien45_vec)
Orien90_locs = np.where((all_stim_label==13))[0]
Orien90_vec = u[Orien90_locs,1:4].mean(0)
Orien90_vec = Orien90_vec/np.linalg.norm(Orien90_vec)
Orien135_locs = np.where((all_stim_label==15))[0]
Orien135_vec = u[Orien135_locs,1:4].mean(0)
Orien135_vec = Orien135_vec/np.linalg.norm(Orien135_vec)
# print different angles.
LR_angle = np.arccos(np.dot(LE_vec,RE_vec))*180/np.pi
HV_angle = np.arccos(np.dot(Orien0_vec,Orien90_vec))*180/np.pi
AO_angle = np.arccos(np.dot(Orien45_vec,Orien135_vec))*180/np.pi
HA_angle = np.arccos(np.dot(Orien0_vec,Orien45_vec))*180/np.pi
Orien_OD_angle = np.arccos(np.dot(LE_vec,Orien0_vec))*180/np.pi
print(f'LE and RE have Angle {LR_angle:.2f}')
print(f'Orien 0 and 90 have Angle {HV_angle:.2f}')
print(f'Orien 45 and 135 have Angle {AO_angle:.2f}')
print(f'Orien 0 and 45 have Angle {HA_angle:.2f}')
print(f'Orien and OD have Angle {Orien_OD_angle:.2f}')
#%% 2.Combine vectors to get OD axis and Orientation plane.
OD_vec = LE_vec*len(LE_locs)-RE_vec*len(RE_locs)
OD_vec = OD_vec/np.linalg.norm(OD_vec)
HV_vec = Orien0_vec*len(Orien0_vec)-Orien90_vec*len(Orien90_vec)
HV_vec = HV_vec/np.linalg.norm(HV_vec)
AO_vec = Orien45_vec*len(Orien45_vec)-Orien135_vec*len(Orien135_vec)
AO_vec = AO_vec/np.linalg.norm(AO_vec)
# calculate all distance of all OD trails to the given axis.
dist_distribution = []
all_LE_vecs = u[LE_locs,1:4]
all_RE_vecs = u[RE_locs,1:4]
all_OD_vecs = np.concatenate([all_LE_vecs,all_RE_vecs],axis = 0)
# all_OD_vecs = u[:,1:4]
for i,c_vec in enumerate(all_OD_vecs):
    c_vec_len = np.linalg.norm(c_vec)
    norm_c_vec = c_vec/c_vec_len
    c_corr = np.dot(norm_c_vec,OD_vec)
    angle = np.arccos(abs(c_corr))*180/np.pi
    dist_distribution.append(angle)
# Orientation Plane fits.
all_Orien_vecs = u[np.where((all_stim_label>8)*(all_stim_label<17))[0],1:4]
#codes below from stack overflow.
tmp_A = []
tmp_b = []
xs = all_Orien_vecs[:,0]
ys = all_Orien_vecs[:,1]
zs = all_Orien_vecs[:,2]
for i in range(len(xs)):
    tmp_A.append([xs[i], ys[i], 1])
    tmp_b.append(zs[i])
b = np.matrix(tmp_b).T
A = np.matrix(tmp_A)
fit = (A.T * A).I * A.T * b
errors = b - A * fit
residual = np.linalg.norm(errors)
orien_norm_vec = np.array([float(fit[0]),float(fit[1]),-1])
orien_norm_vec = orien_norm_vec/np.linalg.norm(orien_norm_vec)
#%% 3. get all stim maps' embedding on given space.
c_OD_map = np.array(ac.OD_t_graphs['OD'].loc['t_value']).reshape(-1, 1)
c_OD_coords = model.transform(c_OD_map.T)
c_OD_vec = c_OD_coords[0,1:4]
c_OD_vec = c_OD_vec/np.linalg.norm(c_OD_vec)
c_HV_map = np.array(ac.Orien_t_graphs['H-V'].loc['t_value']).reshape(-1, 1)
c_HV_coords = model.transform(c_HV_map.T)
c_HV_vec = c_HV_coords[0,1:4]
c_HV_vec = c_HV_vec/np.linalg.norm(c_HV_vec)
c_AO_map = np.array(ac.Orien_t_graphs['A-O'].loc['t_value']).reshape(-1, 1)
c_AO_coords = model.transform(c_AO_map.T)
c_AO_vec = c_AO_coords[0,1:4]
c_AO_vec = c_AO_vec/np.linalg.norm(c_AO_vec)
#%% 4. Plot 3D scatter map with axis.
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
xlim = ax.get_xlim()
ylim = ax.get_ylim()
X,Y = np.meshgrid(np.arange(xlim[0], xlim[1],5),
                  np.arange(ylim[0], ylim[1],5))
Z = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
ax.plot_wireframe(X,Y,Z, color='k')
ax.set_xlabel('PC 2')
ax.set_ylabel('PC 3')
ax.set_zlabel('PC 4')
ax.set_title('Orientation prefered distribution-PCA')
# from Plot_Tools import Arrow3D
arw1 = Arrow3D([0,-OD_vec[0]*20],[0,-OD_vec[1]*20],[0,-OD_vec[2]*20], arrowstyle="->", color="black", lw = 2, mutation_scale=25)
arw1b = Arrow3D([0,-c_OD_vec[0]*20],[0,-c_OD_vec[1]*20],[0,-c_OD_vec[2]*20], arrowstyle="->", color="black", lw = 2, mutation_scale=25,alpha = 0.5)
arw2 = Arrow3D([0,HV_vec[0]*20],[0,HV_vec[1]*20],[0,HV_vec[2]*20], arrowstyle="->", color="red", lw = 2, mutation_scale=25)
arw2b = Arrow3D([0,c_HV_vec[0]*20],[0,c_HV_vec[1]*20],[0,c_HV_vec[2]*20], arrowstyle="->", color="red", lw = 2, mutation_scale=25,alpha = 0.5)
arw3 = Arrow3D([0,AO_vec[0]*20],[0,AO_vec[1]*20],[0,AO_vec[2]*20], arrowstyle="->", color="blue", lw = 2, mutation_scale=25)
arw3b = Arrow3D([0,c_AO_vec[0]*20],[0,c_AO_vec[1]*20],[0,c_AO_vec[2]*20], arrowstyle="->", color="blue", lw = 2, mutation_scale=25,alpha = 0.5)
arw4 = Arrow3D([0,-orien_norm_vec[0]*20],[0,-orien_norm_vec[1]*20],[0,-orien_norm_vec[2]*20], arrowstyle="->", color="yellow", lw = 2, mutation_scale=25)


ax.add_artist(arw1)
ax.add_artist(arw1b)
ax.add_artist(arw2)
ax.add_artist(arw2b)
ax.add_artist(arw3)
ax.add_artist(arw3b)
ax.add_artist(arw4)
fig.tight_layout()
plt.show()
animation = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=150)
animation.save(f'Plot_3D.gif', writer='pillow')


#%% 5. Recover functional map on given axis.
PC2_Comp = comps_stim[1,:]
PC3_Comp = comps_stim[2,:]
PC4_Comp = comps_stim[3,:]
OD_recovered = PC2_Comp*OD_vec[0]+PC3_Comp*OD_vec[1]+PC4_Comp*OD_vec[2]
AO_recovered = PC2_Comp*AO_vec[0]+PC3_Comp*AO_vec[1]+PC4_Comp*AO_vec[2]
HV_recovered = PC2_Comp*HV_vec[0]+PC3_Comp*HV_vec[1]+PC4_Comp*HV_vec[2]
Norm_recovered = PC2_Comp*orien_norm_vec[0]+PC3_Comp*orien_norm_vec[1]+PC4_Comp*orien_norm_vec[2]
OD_explained_var_ratio = explained_var_ratio[1]*OD_vec[0]+explained_var_ratio[2]*OD_vec[1]+explained_var_ratio[3]*OD_vec[2]
HV_explained_var_ratio = explained_var_ratio[1]*HV_vec[0]+explained_var_ratio[2]*HV_vec[1]+explained_var_ratio[3]*HV_vec[2]
AO_explained_var_ratio = explained_var_ratio[1]*AO_vec[0]+explained_var_ratio[2]*AO_vec[1]+explained_var_ratio[3]*AO_vec[2]
print(f'Explained Var Ratio:\nOD:{OD_explained_var_ratio*100:.2f}%,\nHV:{HV_explained_var_ratio*100:.2f}%\nAO:{AO_explained_var_ratio*100:.2f}%')
OD_recovered_map = ac.Generate_Weighted_Cell(OD_recovered)
AO_recovered_map = ac.Generate_Weighted_Cell(AO_recovered)
HV_recovered_map = ac.Generate_Weighted_Cell(HV_recovered)
Norm_recovered_map = ac.Generate_Weighted_Cell(Norm_recovered)
OD_map = ac.Generate_Weighted_Cell(ac.OD_t_graphs['OD'].loc['A_reponse']-ac.OD_t_graphs['OD'].loc['B_response'])
HV_map = ac.Generate_Weighted_Cell(ac.Orien_t_graphs['H-V'].loc['A_reponse']-ac.Orien_t_graphs['H-V'].loc['B_response'])
AO_map = ac.Generate_Weighted_Cell(ac.Orien_t_graphs['A-O'].loc['A_reponse']-ac.Orien_t_graphs['A-O'].loc['B_response'])

Norm_compare = np.hstack([Norm_recovered_map/abs(Norm_recovered_map).max(),OD_map/abs(OD_map).max()])
OD_compare = np.hstack([OD_recovered_map/abs(OD_recovered_map).max(),OD_map/abs(OD_map).max()])
HV_compare = np.hstack([HV_recovered_map/abs(HV_recovered_map).max(),HV_map/abs(HV_map).max()])
AO_compare = np.hstack([AO_recovered_map/abs(AO_recovered_map).max(),AO_map/abs(AO_map).max()])
Norm_compare[:,510:514] = 1
OD_compare[:,510:514] = 1
HV_compare[:,510:514] = 1
AO_compare[:,510:514] = 1

value_max = 1
value_min = -1
font_size = 11
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,5),dpi = 180)
cbar_ax = fig.add_axes([.97, .15, .02, .7])
sns.heatmap(OD_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,0],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(Norm_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,1],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(HV_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,0],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(AO_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,1],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)

axes[0,0].set_title('OD Vector vs OD Map',size = font_size)
axes[0,1].set_title('Orientation Norm Vector vs OD Map',size = font_size)
axes[1,0].set_title('HV Vector vs HV Map',size = font_size)
axes[1,1].set_title('AO Vector vs AO Map',size = font_size)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=None)
fig.tight_layout()
plt.show()