'''
This script will do vector analysis on PCAed data. The aim is to recover functional map and get relationship between maps.

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

    

#%%#################### 1. VECTOR GENERATION##############################
all_spon_pca = ot.Load_Variable(work_path,'All_PCA_Result_Spon.pkl')
used_pc_num = 20
all_loc_vector = pd.DataFrame(columns=['Loc','OD_vec','HV_vec','AO_vec','All_VAR','OD_VAR','HV_VAR','AO_VAR','OD_HV_Angle','OD_AO_Angle','HV_AO_Angle','OD_avr_response','HV_avr_response','AO_avr_response','OD_graph','HV_graph','AO_graph'])
all_loc_vector_color = pd.DataFrame(columns=['Loc','Red_vec','Green_vec','Blue_vec','Red_response','Green_response','Blue_response','Red_graph','Green_graph','Blue_graph'])
for j,cloc in tqdm(enumerate(all_path_dic)):
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    cname = cloc.split('\\')[-1]
    c_pca_result = all_spon_pca[cname] #PC,N_PC*N_dim; embeddings:NSample*N_used_PCNum
    c_pca_model = c_pca_result['PCA_Model']
    c_pc = c_pca_model.components_[:used_pc_num,:] # N_Comp*N_feature
    c_pca_coords = c_pca_model.transform(c_spon.T)[:,:used_pc_num]
    
    #c_pca_coords = c_pca_result['All_embeddings']
    #c_pc = c_pca_result['All_PCs']
    all_tunings = ac.all_cell_tunings
    # Get LE and RE vector.
    LE_vec = np.zeros(used_pc_num)
    RE_vec = np.zeros(used_pc_num)
    Orien0_vec = np.zeros(used_pc_num)
    Orien45_vec = np.zeros(used_pc_num)
    Orien90_vec = np.zeros(used_pc_num)
    Orien135_vec = np.zeros(used_pc_num)
    Red_vec = np.zeros(used_pc_num)
    Green_vec = np.zeros(used_pc_num)
    Blue_vec = np.zeros(used_pc_num)
    for j in range(len(ac.acn)):
        cc_eye = all_tunings.loc['Best_Eye',j+1]
        if cc_eye == 'LE':
            LE_vec += c_pca_coords[j,:]
        elif cc_eye == 'RE':
            RE_vec += c_pca_coords[j,:]
        cc_orien = all_tunings.loc['Best_Orien',j+1]
        if cc_orien == 'Orien0':
            Orien0_vec += c_pca_coords[j,:]
        elif cc_orien == 'Orien45':
            Orien45_vec += c_pca_coords[j,:]
        elif cc_orien == 'Orien90':
            Orien90_vec += c_pca_coords[j,:]
        elif cc_orien == 'Orien135':
            Orien135_vec += c_pca_coords[j,:]
        cc_color = all_tunings.loc['Best_Color',j+1]
        if cc_color == 'Red':
            Red_vec += c_pca_coords[j,:]
        elif cc_color == 'Green':
            Green_vec += c_pca_coords[j,:]
        elif cc_color == 'Blue':
            Blue_vec += c_pca_coords[j,:]

    # Below are test part, using pca will generate sub fig only, so we add vector.
    # LE_vec_response,LE_vec_graph = Vector_AVR_Response(LE_vec,c_pc,c_spon,ac)
    # RE_vec_response,RE_vec_graph = Vector_AVR_Response(RE_vec,c_pc,c_spon,ac)
    # Orien0_vec_response,Orien0_vec_graph = Vector_AVR_Response(Orien0_vec,c_pc,c_spon,ac)
    # Orien90_vec_response,Orien90_vec_graph = Vector_AVR_Response(Orien90_vec,c_pc,c_spon,ac)
    # Orien45_vec_response,Orien45_vec_graph = Vector_AVR_Response(Orien45_vec,c_pc,c_spon,ac)
    # Orien135_vec_response,Orien135_vec_graph = Vector_AVR_Response(Orien135_vec,c_pc,c_spon,ac)
    OD_vec = LE_vec-RE_vec
    HV_vec = Orien0_vec-Orien90_vec
    AO_vec = Orien45_vec-Orien135_vec
    OD_vec = OD_vec/np.linalg.norm(OD_vec)
    HV_vec = HV_vec/np.linalg.norm(HV_vec)
    AO_vec = AO_vec/np.linalg.norm(AO_vec)
    Red_vec = Red_vec/np.linalg.norm(Red_vec)
    Green_vec = Green_vec/np.linalg.norm(Green_vec)
    Blue_vec = Blue_vec/np.linalg.norm(Blue_vec)

    OD_AO_angle,_ = Angle_Calculate(OD_vec,AO_vec)
    OD_HV_angle,_ = Angle_Calculate(OD_vec,HV_vec)
    HV_AO_angle,_ = Angle_Calculate(HV_vec,AO_vec)
    # then calculate var and angle.
    exp_var = np.sum(c_pca_model.explained_variance_ratio_[:used_pc_num])
    OD_var = np.dot(c_pca_model.explained_variance_ratio_[:used_pc_num],(OD_vec*OD_vec))
    AO_var = np.dot(c_pca_model.explained_variance_ratio_[:used_pc_num],(AO_vec*AO_vec))
    HV_var = np.dot(c_pca_model.explained_variance_ratio_[:used_pc_num],(HV_vec*HV_vec))
    OD_AO_angle,_ = Angle_Calculate(OD_vec,AO_vec)
    OD_HV_angle,_ = Angle_Calculate(OD_vec,HV_vec)
    HV_AO_angle,_ = Angle_Calculate(HV_vec,AO_vec)
    # then generate avr response and frame.
    OD_response,OD_recover_map = Vector_AVR_Response(OD_vec,c_pc,c_spon,ac)
    HV_response,HV_recover_map = Vector_AVR_Response(HV_vec,c_pc,c_spon,ac)
    AO_response,AO_recover_map = Vector_AVR_Response(AO_vec,c_pc,c_spon,ac)
    Red_response,Red_recover_map = Vector_AVR_Response(Red_vec,c_pc,c_spon,ac)
    Green_response,Green_recover_map = Vector_AVR_Response(Green_vec,c_pc,c_spon,ac)
    Blue_response,Blue_recover_map = Vector_AVR_Response(Blue_vec,c_pc,c_spon,ac)
    all_loc_vector.loc[len(all_loc_vector),:] = [cname,OD_vec,HV_vec,AO_vec,exp_var,OD_var,HV_var,AO_var,OD_HV_angle,OD_AO_angle,HV_AO_angle,OD_response,HV_response,AO_response,OD_recover_map,HV_recover_map,AO_recover_map]
    all_loc_vector_color.loc[len(all_loc_vector_color),:] = [cname,Red_vec,Green_vec,Blue_vec,Red_response,Green_response,Blue_response,Red_recover_map,Green_recover_map,Blue_recover_map]
ot.Save_Variable(work_path,'All_PCA_Vectors',all_loc_vector)
ot.Save_Variable(work_path,'All_PCA_Vectors_Color',all_loc_vector_color)

#%%#############################2. VECTOR PLOT ######################################
# choose a sample location (L76-18M) as input for visualization.
import colorsys
example_loc = all_loc_vector.loc[7,:]
example_loc_pca = all_spon_pca[example_loc['Loc']]
cloc = all_path_dic[7]
ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
c_tuning = ac.all_cell_tunings
color_files_orien = np.zeros(shape = (len(ac.acn),3))
for i,cc in enumerate(ac.acn):
    cc_orien = c_tuning[cc]['Best_Orien']
    if cc_orien == 'False':
        color_files_orien[i,:] = colorsys.hls_to_rgb(0,0.5,0)
    else:
        c_hue = float(cc_orien[5:])/180
        c_lightness = 0.5
        c_saturation = 1
        color_files_orien[i,:] = colorsys.hls_to_rgb(c_hue, c_lightness, c_saturation)

#%% Plot PC1-3,for od.
from Plot_Tools import Arrow3D
plot_dims = [3,4,5]
mode = 'Orien'
plt.clf()
plt.cla()
elev = 20
azim = 150
u = example_loc_pca['All_embeddings'][:,plot_dims]
fig = plt.figure(figsize = (10,6))
ax = plt.axes(projection='3d')
ax.grid(False)
if mode == 'Orien':
    sc = ax.scatter3D(u[:,0], u[:,1], u[:,2],s = 5,c = color_files_orien)
    # sc = ax.scatter3D(u[:,0], u[:,1], u[:,2],s = 5,c = c_tuning.loc['OD'],cmap = 'bwr')
    color_sets = np.zeros(shape = (8,3))
    for i,c_orien in enumerate(np.arange(0,180,22.5)):
        c_hue = c_orien/180
        c_lightness = 0.5
        c_saturation = 1
        color_sets[i,:] = colorsys.hls_to_rgb(c_hue, c_lightness, c_saturation)
    custom_cmap = mcolors.ListedColormap(color_sets)
    cax = fig.add_axes([0.85, 0.3, 0.03, 0.4])
    bounds = np.arange(0,202.5,22.5) # bound will have 1 more unit after graph.
    norm = mpl.colors.BoundaryNorm(bounds, custom_cmap.N)
    c_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=custom_cmap),cax=cax, label='Prefer Orientation')
    c_bar.set_ticks(np.arange(0,180,22.5)+11.25)
    c_bar.set_ticklabels(np.arange(0,180,22.5))
    c_bar.ax.tick_params(size=0)
elif mode == 'OD':
    sc = ax.scatter3D(u[:,0], u[:,1], u[:,2],s = 5,c = c_tuning.loc['OD'],cmap = 'bwr')
    cbar = fig.colorbar(sc, shrink=0.5)
    cbar.set_label('OD Tuning')

ax.set_xlabel('PC 4')
ax.set_ylabel('PC 5')
ax.set_zlabel('PC 6')
ax.set_title('Orientation Preference distribution-PCA')
ax.view_init(elev=elev, azim=azim)
OD_vec_plot = example_loc['OD_vec'][plot_dims]
HV_vec_plot = example_loc['HV_vec'][plot_dims]
AO_vec_plot = example_loc['AO_vec'][plot_dims]
OD_vec_plot = OD_vec_plot/np.linalg.norm(example_loc['OD_vec'])
HV_vec_plot = HV_vec_plot/np.linalg.norm(example_loc['HV_vec'])
AO_vec_plot = AO_vec_plot/np.linalg.norm(example_loc['AO_vec'])

arw_od = Arrow3D([0,-OD_vec_plot[0]*30],[0,-OD_vec_plot[1]*30],[0,-OD_vec_plot[2]*30], arrowstyle="->", color="black", lw = 2, mutation_scale=15)
arw_hv = Arrow3D([0,-HV_vec_plot[0]*30],[0,-HV_vec_plot[1]*30],[0,-HV_vec_plot[2]*30], arrowstyle="->", color="blue", lw = 2, mutation_scale=15)
arw_ao = Arrow3D([0,-AO_vec_plot[0]*30],[0,-AO_vec_plot[1]*30],[0,-AO_vec_plot[2]*30], arrowstyle="->", color="yellow", lw = 2, mutation_scale=15)
ax.add_artist(arw_od)
ax.add_artist(arw_hv)
ax.add_artist(arw_ao)
fig.tight_layout()
def update(frame):
    ax.view_init(elev=25, azim=frame)  # Update the view angle for each frame
    return ax,
animation = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=150)
animation.save(f'{work_path}\\PC4_6_Orien_Vecs.gif', writer='pillow')
#%%###################### 3. Compare Functional Maps ###################################
real_od_response = np.array(ac.OD_t_graphs['OD'].loc['t_value'])
real_hv_response = np.array(ac.Orien_t_graphs['H-V'].loc['t_value'])
real_ao_response = np.array(ac.Orien_t_graphs['A-O'].loc['t_value'])
real_red_response = np.array(ac.Color_t_graphs['Red-White'].loc['t_value'])
real_green_response = np.array(ac.Color_t_graphs['Green-White'].loc['t_value'])
real_blue_response = np.array(ac.Color_t_graphs['Blue-White'].loc['t_value'])
od_map = ac.Generate_Weighted_Cell(real_od_response)
od_map = od_map/od_map.max()
od_recover_map = example_loc['OD_graph']/example_loc['OD_graph'].max()
od_compare = np.hstack((od_map,od_recover_map))
od_similarity,_ = stats.pearsonr(real_od_response,example_loc['OD_avr_response'])

hv_map = ac.Generate_Weighted_Cell(real_hv_response)
hv_map = hv_map/hv_map.max()
hv_recover_map = example_loc['HV_graph']/example_loc['HV_graph'].max()
hv_compare = np.hstack((hv_map,hv_recover_map))
hv_similarity,_ = stats.pearsonr(real_hv_response,example_loc['HV_avr_response'])

ao_map = ac.Generate_Weighted_Cell(real_ao_response)
ao_map = ao_map/ao_map.max()
ao_recover_map = example_loc['AO_graph']/example_loc['AO_graph'].max()
ao_compare = np.hstack((ao_map,ao_recover_map))
ao_similarity,_ = stats.pearsonr(real_ao_response,example_loc['AO_avr_response'])

od_compare[:,510:514] = 10
hv_compare[:,510:514] = 10
ao_compare[:,510:514] = 10
#%%
plt.clf()
plt.cla()
value_max = 1
value_min = -1
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(4,6),dpi = 180)
cbar_ax = fig.add_axes([0.95, .35, .03, .3])
sns.heatmap(od_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(hv_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(ao_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[2],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
axes[0].set_title('OD Map                            OD Axis',size = 8)
axes[1].set_title('HV Map                            HV Axis',size = 8)
axes[2].set_title('AO Map                            AO Axis',size = 8)
fig.suptitle('PCA Vector Represent Function Maps',size = 10)
fig.tight_layout()
plt.show()

#%%######################## 4.VECTOR STATS ###########################################
#%% 1. Angle between different axes
angle_frame_plot = pd.DataFrame(columns=['Loc','Angle','Axes'])
for i in range(len(all_loc_vector)):
    cloc_info = all_loc_vector.loc[i]
    angle_frame_plot.loc[len(angle_frame_plot)] = [cloc_info['Loc'],cloc_info['OD_HV_Angle'],'OD_HV']
    angle_frame_plot.loc[len(angle_frame_plot)] = [cloc_info['Loc'],cloc_info['OD_AO_Angle'],'OD_AO']
    angle_frame_plot.loc[len(angle_frame_plot)] = [cloc_info['Loc'],cloc_info['HV_AO_Angle'],'HV_AO']
plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5,4),dpi = 180)
sns.boxplot(data = angle_frame_plot,x = 'Axes',y = 'Angle',width = 0.5,ax = ax)
ax.axhline(y=90, color='gray', linestyle='--')
ax.set(ylim = (0,140))
fig.suptitle('Angle Between Axes')
fig.tight_layout()
plt.show()
#%% 2. Vector vs Stimmap Similarity
similarity_frame = pd.DataFrame(columns=['Loc','Pearson R','Map Name'])
for i in tqdm(range(len(all_path_dic))):
    c_ac = ot.Load_Variable_v2(all_path_dic[i],'Cell_Class.pkl')
    real_od_response = c_ac.OD_t_graphs['OD'].loc['t_value']
    real_hv_response = c_ac.Orien_t_graphs['H-V'].loc['t_value']
    real_ao_response = c_ac.Orien_t_graphs['A-O'].loc['t_value']
    c_vecs = all_loc_vector.loc[i]
    c_od_response = c_vecs['OD_avr_response']
    c_hv_response = c_vecs['HV_avr_response']
    c_ao_response = c_vecs['AO_avr_response']
    cname = all_path_dic[i].split('\\')[-1]
    od_r,_ = stats.pearsonr(real_od_response,c_od_response)
    hv_r,_ = stats.pearsonr(real_hv_response,c_hv_response)
    ao_r,_ = stats.pearsonr(real_ao_response,c_ao_response)
    similarity_frame.loc[len(similarity_frame),:] = [cname,od_r,'OD']
    similarity_frame.loc[len(similarity_frame),:] = [cname,hv_r,'HV']
    similarity_frame.loc[len(similarity_frame),:] = [cname,ao_r,'AO']

    c_vecs_color = all_loc_vector_color.loc[i]
    c_red_response = c_vecs_color['Red_response']
    c_green_response = c_vecs_color['Green_response']
    c_blue_response = c_vecs_color['Blue_response']
    real_red_response = c_ac.Color_t_graphs['Red-White'].loc['t_value']
    real_green_response = c_ac.Color_t_graphs['Green-White'].loc['t_value']
    real_blue_response = c_ac.Color_t_graphs['Blue-White'].loc['t_value']
    red_r,_ = stats.pearsonr(real_red_response,c_red_response)
    green_r,_ = stats.pearsonr(real_green_response,c_green_response)
    blue_r,_ = stats.pearsonr(real_blue_response,c_blue_response)
    similarity_frame.loc[len(similarity_frame),:] = [cname,red_r,'Red']
    similarity_frame.loc[len(similarity_frame),:] = [cname,green_r,'Green']
    similarity_frame.loc[len(similarity_frame),:] = [cname,blue_r,'Blue']
    

# Plot Vector Stimmap Similarity
plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3,6),dpi = 180)
sns.boxplot(data = similarity_frame,x = 'Map Name',y = 'Pearson R',width = 0.5,ax = ax, showfliers=False)
# ax.axhline(y=90, color='gray', linestyle='--')
# ax.set(ylim = (0,1))
fig.suptitle('Similarity of Functional Axes',size = 14)
fig.tight_layout()
plt.show()

