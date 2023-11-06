'''
This is the whole result of all data point PCA stats.
All PCA results are here.

'''


#%% Import
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
from Advanced_Tools import *

work_path = r'D:\_Path_For_Figs\S1_Stat_PCA'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Datas_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)

#%%##########################PCA Stim Graphs#########################################
all_PCA_results = {}
comp_limit = 20 # how many comps are used on analysisi below？
for i,cc in tqdm(enumerate(all_path_dic)):
    cloc_name = cc.split('\\')[-1]
    all_PCA_results[cloc_name]={}
    c_ac = ot.Load_Variable_v2(cc,'Cell_Class.pkl')
    all_stim_frame,all_stim_label = c_ac.Combine_Frame_Labels(color = True)
    c_comps_stim,c_coords_stim,c_model = Z_PCA(Z_frame=all_stim_frame,sample='Frame')
    all_PCA_results[cloc_name]['PC_Axes'] = c_comps_stim[:comp_limit,:]
    all_PCA_results[cloc_name]['Frame_Coords'] = c_coords_stim[:,:comp_limit]
    all_PCA_results[cloc_name]['PCA_Model'] = c_model
    all_PCA_results[cloc_name]['Top20_VAR'] = c_model.explained_variance_ratio_[:comp_limit].sum()
    all_PCA_results[cloc_name]['PC1_VAR'] = c_model.explained_variance_ratio_[0]
    LE_locs = np.where((all_stim_label>0)*(all_stim_label<9)*(all_stim_label%2 == 1))[0]
    LE_vecs = c_coords_stim[LE_locs,:comp_limit]
    RE_locs = np.where((all_stim_label>0)*(all_stim_label<9)*(all_stim_label%2 == 0))[0]
    RE_vecs = c_coords_stim[RE_locs,:comp_limit]
    Orien0_locs = np.where((all_stim_label==9))[0]
    Orien0_vecs = c_coords_stim[Orien0_locs,:comp_limit]
    Orien45_locs = np.where((all_stim_label==11))[0]
    Orien45_vecs = c_coords_stim[Orien45_locs,:comp_limit]
    Orien90_locs = np.where((all_stim_label==13))[0]
    Orien90_vecs = c_coords_stim[Orien90_locs,:comp_limit]
    Orien135_locs = np.where((all_stim_label==15))[0]
    Orien135_vecs = c_coords_stim[Orien135_locs,:comp_limit]
    all_PCA_results[cloc_name]['LE_vec'] = LE_vecs.mean(0)
    all_PCA_results[cloc_name]['RE_vec'] = RE_vecs.mean(0)
    all_PCA_results[cloc_name]['OD_vec'] = np.vstack([LE_vecs,-RE_vecs]).mean(0)
    all_PCA_results[cloc_name]['Orien0_vec'] = Orien0_vecs.mean(0)
    all_PCA_results[cloc_name]['Orien45_vec'] = Orien45_vecs.mean(0)
    all_PCA_results[cloc_name]['Orien90_vec'] = Orien90_vecs.mean(0)
    all_PCA_results[cloc_name]['Orien135_vec'] = Orien135_vecs.mean(0)
    all_PCA_results[cloc_name]['HV_vec'] = np.vstack([Orien0_vecs,-Orien90_vecs]).mean(0)
    all_PCA_results[cloc_name]['AO_vec'] = np.vstack([Orien45_vecs,-Orien135_vecs]).mean(0)
    # basic fit, including all orientation vectors and plane parameter.
    all_Orien_locs = np.where((all_stim_label>8)*(all_stim_label<17))[0]
    all_Orien_vecs = c_coords_stim[all_Orien_locs,:comp_limit]
    #codes below from stack overflow.
    tmp_A = []
    tmp_b = []
    xs = all_Orien_vecs[:,1]
    ys = all_Orien_vecs[:,2]
    zs = all_Orien_vecs[:,3]
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
    all_PCA_results[cloc_name]['Orien_Plane_Fit'] = fit # z = ax+by+c
    all_PCA_results[cloc_name]['Orien_Plane_Norm'] = orien_norm_vec
ot.Save_Variable(work_path,'All_PCA_Informations',all_PCA_results)
#%% #######################AXIS ANALYSIS##############################################
# This part will generate axis analyze on all 19 axis (Norm only on 3 axis), calculate angle distribution,recovered map(with similarity to stim map), 
axis_path = work_path+r'\All_Stim_Axis'
ot.mkdir(axis_path)
all_ax_similarity = pd.DataFrame(columns=['Loc','Axes','Corr','Vector','Avr_Response','Avr_Graph']) # only PC2-4 used here.
all_ax_angle = pd.DataFrame(columns=['Loc','Axes_A','Axes_B','Angle','Cosine'])
for i,cc in tqdm(enumerate(all_path_dic)):
    # prepare data and 
    cloc_name = cc.split('\\')[-1]
    c_ac = ot.Load_Variable_v2(cc,'Cell_Class.pkl')
    real_OD_response = c_ac.OD_t_graphs['OD'].loc['A_reponse']-c_ac.OD_t_graphs['OD'].loc['B_response']
    real_HV_response = c_ac.Orien_t_graphs['H-V'].loc['A_reponse']-c_ac.Orien_t_graphs['H-V'].loc['B_response']
    real_AO_response = c_ac.Orien_t_graphs['A-O'].loc['A_reponse']-c_ac.Orien_t_graphs['A-O'].loc['B_response']
    real_OD_response = real_OD_response/abs(real_OD_response).max()
    real_HV_response = real_HV_response/abs(real_HV_response).max()
    real_AO_response = real_AO_response/abs(real_AO_response).max()
    real_OD_map = c_ac.Generate_Weighted_Cell(real_OD_response)
    real_HV_map = c_ac.Generate_Weighted_Cell(real_HV_response)
    real_AO_map = c_ac.Generate_Weighted_Cell(real_AO_response)

    # step 1, calculate all recovered map.
    c_OD_ax = all_PCA_results[cloc_name]['OD_vec'][1:]
    # recovered_OD_response = c_OD_ax[0]*all_PCA_results[cloc_name]['PC_Axes'][1,:]+c_OD_ax[1]*all_PCA_results[cloc_name]['PC_Axes'][2,:]+c_OD_ax[2]*all_PCA_results[cloc_name]['PC_Axes'][3,:]
    recovered_OD_response = np.dot(c_OD_ax,all_PCA_results[cloc_name]['PC_Axes'][1:,:])
    recovered_OD_map = c_ac.Generate_Weighted_Cell(recovered_OD_response)
    recovered_OD_map = recovered_OD_map/abs(recovered_OD_map.max())

    c_HV_ax = all_PCA_results[cloc_name]['HV_vec'][1:]
    # recovered_HV_response = c_HV_ax[0]*all_PCA_results[cloc_name]['PC_Axes'][1,:]+c_HV_ax[1]*all_PCA_results[cloc_name]['PC_Axes'][2,:]+c_HV_ax[2]*all_PCA_results[cloc_name]['PC_Axes'][3,:]
    recovered_HV_response = np.dot(c_HV_ax,all_PCA_results[cloc_name]['PC_Axes'][1:,:])
    recovered_HV_map = c_ac.Generate_Weighted_Cell(recovered_HV_response)
    recovered_HV_map = recovered_HV_map/abs(recovered_HV_map.max())

    c_AO_ax = all_PCA_results[cloc_name]['AO_vec'][1:]
    # recovered_AO_response = c_AO_ax[0]*all_PCA_results[cloc_name]['PC_Axes'][1,:]+c_AO_ax[1]*all_PCA_results[cloc_name]['PC_Axes'][2,:]+c_AO_ax[2]*all_PCA_results[cloc_name]['PC_Axes'][3,:]
    recovered_AO_response = np.dot(c_AO_ax,all_PCA_results[cloc_name]['PC_Axes'][1:,:])
    recovered_AO_map = c_ac.Generate_Weighted_Cell(recovered_AO_response)
    recovered_AO_map = recovered_AO_map/abs(recovered_AO_map.max())
    ## Addemtent, use cross of HV&AO to get real real norm vecs.
    # c_Norm_ax = all_PCA_results[cloc_name]['Orien_Plane_Norm']
    # recovered_Norm_response = c_Norm_ax[0]*all_PCA_results[cloc_name]['PC_Axes'][1,:]+c_Norm_ax[1]*all_PCA_results[cloc_name]['PC_Axes'][2,:]+c_Norm_ax[2]*all_PCA_results[cloc_name]['PC_Axes'][3,:]
    # recovered_Norm_map = c_ac.Generate_Weighted_Cell(recovered_Norm_response)
    # recovered_Norm_map = recovered_Norm_map/abs(recovered_Norm_map.max())
    real_norm = np.cross(c_HV_ax[:3],c_AO_ax[:3])
    recovered_Norm_response = real_norm[0]*all_PCA_results[cloc_name]['PC_Axes'][1,:]+real_norm[1]*all_PCA_results[cloc_name]['PC_Axes'][2,:]+real_norm[2]*all_PCA_results[cloc_name]['PC_Axes'][3,:]
    recovered_Norm_map = c_ac.Generate_Weighted_Cell(recovered_Norm_response)
    recovered_Norm_map = recovered_Norm_map/abs(recovered_Norm_map.max())
    # get compare map and save recovered graph above in a file.
    OD_compare = np.hstack([recovered_OD_map,real_OD_map])
    AO_compare = np.hstack([recovered_AO_map,real_AO_map])
    HV_compare = np.hstack([recovered_HV_map,real_HV_map])
    Norm_compare = np.hstack([recovered_Norm_map,real_OD_map])
    OD_compare[:,510:514]=1
    HV_compare[:,510:514] = 1
    AO_compare[:,510:514] = 1
    Norm_compare[:,510:514] = 1

    plt.clf()
    plt.cla()
    value_max = 1
    value_min = -1
    font_size = 11
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11,6),dpi = 180)
    cbar_ax = fig.add_axes([.92, .15, .02, .7])
    sns.heatmap(OD_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,0],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    sns.heatmap(Norm_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,1],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    sns.heatmap(HV_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,0],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    sns.heatmap(AO_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,1],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    axes[0,0].set_title('OD Vector vs OD Map',size = font_size)
    axes[0,1].set_title('Orientation Norm Vector vs OD Map',size = font_size)
    axes[1,0].set_title('HV Vector vs HV Map',size = font_size)
    axes[1,1].set_title('AO Vector vs AO Map',size = font_size)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=None)
    # fig.tight_layout()
    fig.figure.savefig(f'{axis_path}\\Axis_Compare_{cloc_name}.png')
    # step2, get compare map between ax and stim map.
    od_sim,_ = stats.pearsonr(recovered_OD_response,np.array(real_OD_response))
    hv_sim,_ = stats.pearsonr(recovered_HV_response,np.array(real_HV_response))
    ao_sim,_ = stats.pearsonr(recovered_AO_response,np.array(real_AO_response))
    norm_sim,_ = stats.pearsonr(recovered_Norm_response,np.array(real_OD_response))
    all_ax_similarity.loc[len(all_ax_similarity),:] = [cloc_name,'OD_vec',od_sim,c_OD_ax,recovered_OD_response,recovered_OD_map]
    all_ax_similarity.loc[len(all_ax_similarity),:] = [cloc_name,'Norm_vec',norm_sim,real_norm,recovered_Norm_response,recovered_Norm_map]
    all_ax_similarity.loc[len(all_ax_similarity),:] = [cloc_name,'HV_vec',hv_sim,c_HV_ax,recovered_HV_response,recovered_HV_map]
    all_ax_similarity.loc[len(all_ax_similarity),:] = [cloc_name,'AO_vec',ao_sim,c_AO_ax,recovered_AO_response,recovered_AO_map]
    # step3, calculate angle axis. pair wised.
    #'Loc','Axes_A','Axes_B','Angle','Cosine'
    OD_norm_cos = np.dot(c_OD_ax[:3]/np.linalg.norm(c_OD_ax[:3]),real_norm/np.linalg.norm(real_norm))
    OD_HV_cos = np.dot(c_OD_ax/np.linalg.norm(c_OD_ax),c_HV_ax/np.linalg.norm(c_HV_ax))
    OD_AO_cos = np.dot(c_OD_ax/np.linalg.norm(c_OD_ax),c_AO_ax/np.linalg.norm(c_AO_ax))
    HV_AO_cos = np.dot(c_HV_ax/np.linalg.norm(c_HV_ax),c_AO_ax/np.linalg.norm(c_AO_ax))
    OD_Norm_angle = np.arccos(OD_norm_cos)*180/np.pi
    OD_HV_angle = np.arccos(OD_HV_cos)*180/np.pi
    OD_AO_angle = np.arccos(OD_AO_cos)*180/np.pi
    HV_AO_angle = np.arccos(HV_AO_cos)*180/np.pi
    all_ax_angle.loc[len(all_ax_angle),:] = [cloc_name,'HV','AO',HV_AO_angle,HV_AO_cos]
    all_ax_angle.loc[len(all_ax_angle),:] = [cloc_name,'OD','AO',OD_AO_angle,OD_AO_cos]
    all_ax_angle.loc[len(all_ax_angle),:] = [cloc_name,'OD','HV',OD_HV_angle,OD_HV_cos]
    all_ax_angle.loc[len(all_ax_angle),:] = [cloc_name,'OD','Norm',OD_Norm_angle,OD_norm_cos]

ot.Save_Variable(work_path,'All_Ax_Angle',all_ax_angle)
ot.Save_Variable(work_path,'All_Ax_Similarity',all_ax_similarity)
#%% Stats all angles seperately, test verticals.
angle_stats = pd.DataFrame(0,index = range(8),columns=['HV-AO','OD-HV','OD-AO'])
hv_ao_angles = np.array(all_ax_angle.groupby('Axes_A').get_group('HV')['Angle'])
od_ao_angles = np.array(all_ax_angle.groupby('Axes_A').get_group('OD').groupby('Axes_B').get_group('AO')['Angle'])
od_hv_angles = np.array(all_ax_angle.groupby('Axes_A').get_group('OD').groupby('Axes_B').get_group('HV')['Angle'])
angle_stats['HV-AO'] = hv_ao_angles
angle_stats['OD-HV'] = od_hv_angles
angle_stats['OD-AO'] = od_ao_angles
angle_stats_m = pd.melt(angle_stats,var_name='Axes',value_name='Angle')
plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3,4),dpi = 180)
ax.axhline(y=90, color='gray', linestyle='--')
sns.boxplot(data = angle_stats_m,x = 'Axes',y = 'Angle',width=0.3,ax = ax)
ax.set(ylim = (60,110))
ax.set_title('Angle of Different Axes')
fig.tight_layout()
plt.show()
#%% stats all ax similarities.
used_ax_similarity = all_ax_similarity[all_ax_similarity['Axes']!='Norm_vec']
plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3,4),dpi = 180)
# ax.axhline(y=90, color='gray', linestyle='--')
sns.boxplot(data = used_ax_similarity,x = 'Axes',y = 'Corr',width=0.3,showfliers = False,ax = ax)
ax.set(ylim = (0.9,1))
ax.set_title('Axes Similarity to Stim-Map')
ax.set_xticklabels(['OD','HV','AO'])
fig.tight_layout()
plt.show()
#%%#########################SVM SEPERATOR################################
# This part will do svm classifier on spon data using pca models.
all_loc_scores = np.zeros(len(all_path_dic))
all_recovered_series = {}
all_repeat_similar_frame = pd.DataFrame(columns=['Loc','ID','Frame_Count','Event_Count','Corr','Averaged_Response'])
all_repeat_freq_frame = pd.DataFrame(columns=['Loc','Network','Event_num','Frame_num','Freq','Total_Frame_Num'])
all_repeat_map_similar_frame = pd.DataFrame(columns=['Loc','Map','Corr','Recovered_Response'])
for i,cc in tqdm(enumerate(all_path_dic)):
    # prepare data and 
    cloc_name = cc.split('\\')[-1]
    c_ac = ot.Load_Variable_v2(cc,'Cell_Class.pkl')
    c_spon = np.array(ot.Load_Variable(cc,'Spon_Before.pkl'))
    N = 100 # shuffle times.
    c_model = all_PCA_results[cloc_name]['PCA_Model']
    all_stim_frame,all_stim_label = c_ac.Combine_Frame_Labels(color = True)
    all_stim_embeddings = c_model.transform(all_stim_frame)[:,:20]
    all_spon_embeddings = c_model.transform(c_spon)[:,:20]
    classifier,score = SVM_Classifier(all_stim_embeddings,all_stim_label,C = 10)
    all_loc_scores[i] = score
    predicted_spon_frame = SVC_Fit(classifier=classifier,data = all_spon_embeddings,thres_prob=0)
    all_recovered_response = list(set(predicted_spon_frame))
    all_recovered_series[cloc_name] = predicted_spon_frame
    # get each id recovered map
    for j in range(1,23):
        if j in all_recovered_response: # meaning we have this id 
            
            c_repeat_series = predicted_spon_frame==j
            c_repeat_framecount = c_repeat_series.sum()
            c_repeat_eventcount = Event_Counter(c_repeat_series)
            all_repeat_loc = np.where(c_repeat_series)[0]
            c_template = all_stim_frame[all_stim_label==j].mean(0)
            c_recover = c_spon[all_repeat_loc,:].mean(0)
            corr,_ = stats.pearsonr(c_template,c_recover)
            all_repeat_similar_frame.loc[len(all_repeat_similar_frame),:]=[cloc_name,j,c_repeat_framecount,c_repeat_eventcount,corr,c_recover]
        else:
            all_repeat_similar_frame.loc[len(all_repeat_similar_frame),:]=[cloc_name,j,0,0,0,np.zeros(c_spon.shape[1])]
    # get all 3 network repeat times
    spon_len = len(predicted_spon_frame)
    all_Eye_repeats = (predicted_spon_frame>0)*(predicted_spon_frame<9)
    all_Orien_repeats = (predicted_spon_frame>8)*(predicted_spon_frame<17)
    all_Color_repeats = (predicted_spon_frame>16)*(predicted_spon_frame<23)
    all_Eye_repeat_events = Event_Counter(all_Eye_repeats)
    all_Orien_repeat_events = Event_Counter(all_Orien_repeats)
    all_Color_repeat_events = Event_Counter(all_Color_repeats)
    all_repeat_freq_frame.loc[len(all_repeat_freq_frame),:] = [cloc_name,'Eye',all_Eye_repeat_events,all_Eye_repeats.sum(),all_Eye_repeat_events*1.301/spon_len,spon_len]
    all_repeat_freq_frame.loc[len(all_repeat_freq_frame),:] = [cloc_name,'Orientation',all_Orien_repeat_events,all_Orien_repeats.sum(),all_Orien_repeat_events*1.301/spon_len,spon_len]
    all_repeat_freq_frame.loc[len(all_repeat_freq_frame),:] = [cloc_name,'Color',all_Color_repeat_events,all_Color_repeats.sum(),all_Color_repeat_events*1.301/spon_len,spon_len]
    
    # get each avr graphs.
    c_loc_graphs = all_repeat_similar_frame.groupby('Loc').get_group(cloc_name)
    def Weight_Avr(all_graphs,id_lists):
        c_cond_count = 0
        recovered_frame = np.zeros(c_spon.shape[1],dtype='f8')
        for j,c_cond in enumerate(id_lists):
            c_id_response = all_graphs[all_graphs['ID']==c_cond].iloc[0,:]
            c_cond_count += c_id_response.loc['Frame_Count']
            recovered_frame += c_id_response['Frame_Count']*c_id_response['Averaged_Response']
        if recovered_frame.sum() != 0: # avoid non-response.
            recovered_frame = recovered_frame/abs(recovered_frame).max()
        return recovered_frame
    
    c_RE_response = Weight_Avr(c_loc_graphs,[2,4,6,8])
    c_LE_response = Weight_Avr(c_loc_graphs,[1,3,5,7])
    c_Orien0_response = Weight_Avr(c_loc_graphs,[9])
    c_Orien45_response = Weight_Avr(c_loc_graphs,[11])
    c_Orien90_response = Weight_Avr(c_loc_graphs,[13])
    c_Orien135_response = Weight_Avr(c_loc_graphs,[15])
    c_Red_response = Weight_Avr(c_loc_graphs,[17])
    c_Green_response = Weight_Avr(c_loc_graphs,[19])
    c_Blue_response = Weight_Avr(c_loc_graphs,[21])
    # and generate maps.
    c_RE_map = c_ac.Generate_Weighted_Cell(c_RE_response)
    c_LE_map = c_ac.Generate_Weighted_Cell(c_LE_response)
    c_Orien0_map = c_ac.Generate_Weighted_Cell(c_Orien0_response)
    c_Orien45_map = c_ac.Generate_Weighted_Cell(c_Orien45_response)
    c_Orien90_map = c_ac.Generate_Weighted_Cell(c_Orien90_response)
    c_Orien135_map = c_ac.Generate_Weighted_Cell(c_Orien135_response)
    c_Red_map = c_ac.Generate_Weighted_Cell(c_Red_response)
    c_Green_map = c_ac.Generate_Weighted_Cell(c_Green_response)
    c_Blue_map = c_ac.Generate_Weighted_Cell(c_Blue_response)
    # and real maps here.
    real_LE_map = c_ac.Generate_Weighted_Cell(c_ac.OD_t_graphs['L-0'].loc['t_value'])
    real_RE_map = c_ac.Generate_Weighted_Cell(c_ac.OD_t_graphs['R-0'].loc['t_value'])
    real_Orien0_map = c_ac.Generate_Weighted_Cell(c_ac.Orien_t_graphs['Orien0-0'].loc['t_value'])
    real_Orien45_map = c_ac.Generate_Weighted_Cell(c_ac.Orien_t_graphs['Orien45-0'].loc['t_value'])
    real_Orien90_map = c_ac.Generate_Weighted_Cell(c_ac.Orien_t_graphs['Orien90-0'].loc['t_value'])
    real_Orien135_map = c_ac.Generate_Weighted_Cell(c_ac.Orien_t_graphs['Orien135-0'].loc['t_value'])
    real_Red_map = c_ac.Generate_Weighted_Cell(c_ac.Color_t_graphs['Red-0'].loc['t_value'])
    real_Green_map = c_ac.Generate_Weighted_Cell(c_ac.Color_t_graphs['Green-0'].loc['t_value'])
    real_Blue_map = c_ac.Generate_Weighted_Cell(c_ac.Color_t_graphs['Blue-0'].loc['t_value'])
    real_LE_map = real_LE_map/abs(real_LE_map).max()
    real_RE_map = real_RE_map/abs(real_RE_map).max()
    real_Orien0_map = real_Orien0_map/abs(real_Orien0_map).max()
    real_Orien45_map = real_Orien45_map/abs(real_Orien45_map).max()
    real_Orien90_map = real_Orien90_map/abs(real_Orien90_map).max()
    real_Orien135_map = real_Orien135_map/abs(real_Orien135_map).max()
    real_Red_map = real_Red_map/abs(real_Red_map).max()
    real_Green_map = real_Green_map/abs(real_Green_map).max()
    real_Blue_map = real_Blue_map/abs(real_Blue_map).max()
    # get all corrs
    corr_LE,_ = stats.pearsonr(c_LE_response,np.array(c_ac.OD_t_graphs['L-0'].loc['t_value']))
    corr_RE,_ = stats.pearsonr(c_RE_response,np.array(c_ac.OD_t_graphs['R-0'].loc['t_value']))
    corr_orien0,_ = stats.pearsonr(c_Orien0_response,np.array(c_ac.Orien_t_graphs['Orien0-0'].loc['t_value']))
    corr_orien45,_ = stats.pearsonr(c_Orien45_response,np.array(c_ac.Orien_t_graphs['Orien45-0'].loc['t_value']))
    corr_orien90,_ = stats.pearsonr(c_Orien90_response,np.array(c_ac.Orien_t_graphs['Orien90-0'].loc['t_value']))
    corr_orien135,_ = stats.pearsonr(c_Orien135_response,np.array(c_ac.Orien_t_graphs['Orien135-0'].loc['t_value']))
    corr_red,_ = stats.pearsonr(c_Red_response,np.array(c_ac.Color_t_graphs['Red-0'].loc['t_value']))
    corr_green,_ = stats.pearsonr(c_Green_response,np.array(c_ac.Color_t_graphs['Green-0'].loc['t_value']))
    corr_blue,_ = stats.pearsonr(c_Blue_response,np.array(c_ac.Color_t_graphs['Blue-0'].loc['t_value']))
    #Save maps. ['Loc','Map','Corr','Recovered_Response']
    all_repeat_map_similar_frame.loc[len(all_repeat_map_similar_frame),:]=[cloc_name,'LE',corr_LE,c_LE_response]
    all_repeat_map_similar_frame.loc[len(all_repeat_map_similar_frame),:]=[cloc_name,'RE',corr_RE,c_RE_response]
    all_repeat_map_similar_frame.loc[len(all_repeat_map_similar_frame),:]=[cloc_name,'Orien0',corr_orien0,c_Orien0_response]
    all_repeat_map_similar_frame.loc[len(all_repeat_map_similar_frame),:]=[cloc_name,'Orien45',corr_orien45,c_Orien45_response]
    all_repeat_map_similar_frame.loc[len(all_repeat_map_similar_frame),:]=[cloc_name,'Orien90',corr_orien90,c_Orien90_response]
    all_repeat_map_similar_frame.loc[len(all_repeat_map_similar_frame),:]=[cloc_name,'Orien135',corr_orien135,c_Orien135_response]
    all_repeat_map_similar_frame.loc[len(all_repeat_map_similar_frame),:]=[cloc_name,'Red',corr_red,c_Red_response]
    all_repeat_map_similar_frame.loc[len(all_repeat_map_similar_frame),:]=[cloc_name,'Green',corr_green,c_Green_response]
    all_repeat_map_similar_frame.loc[len(all_repeat_map_similar_frame),:]=[cloc_name,'Blue',corr_blue,c_Blue_response]
    # plot 9*9 graphs.
    LE_compare = np.hstack([c_LE_map,real_LE_map])
    LE_compare[:,510:514]=1
    RE_compare = np.hstack([c_RE_map,real_RE_map])
    RE_compare[:,510:514]=1
    Orien0_compare = np.hstack([c_Orien0_map,real_Orien0_map])
    Orien0_compare[:,510:514]=1
    Orien45_compare = np.hstack([c_Orien45_map,real_Orien45_map])
    Orien45_compare[:,510:514]=1
    Orien90_compare = np.hstack([c_Orien90_map,real_Orien90_map])
    Orien90_compare[:,510:514]=1
    Orien135_compare = np.hstack([c_Orien135_map,real_Orien135_map])
    Orien135_compare[:,510:514]=1
    Red_compare = np.hstack([c_Red_map,real_Red_map])
    Red_compare[:,510:514]=1
    Green_compare = np.hstack([c_Green_map,real_Green_map])
    Green_compare[:,510:514]=1
    Blue_compare = np.hstack([c_Blue_map,real_Blue_map])
    Blue_compare[:,510:514]=1
    plt.clf()
    plt.cla()
    value_max = 1
    value_min = -1
    font_size = 11
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(22,10),dpi = 180)
    cbar_ax = fig.add_axes([.96, .15, .01, .7])
    sns.heatmap(LE_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,0],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    sns.heatmap(RE_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,0],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    sns.heatmap(Orien0_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,1],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    sns.heatmap(Orien45_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,2],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    sns.heatmap(Orien90_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,1],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    sns.heatmap(Orien135_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,2],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    sns.heatmap(Red_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[2,0],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    sns.heatmap(Green_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[2,1],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    sns.heatmap(Blue_compare,center = 0,xticklabels=False,yticklabels=False,ax = axes[2,2],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    axes[0,0].set_title('Left Eye Compare',size = font_size)
    axes[1,0].set_title('Right Eye Compare',size = font_size)
    axes[0,1].set_title('Orientation 0 Compare',size = font_size)
    axes[0,2].set_title('Orientation 45 Compare',size = font_size)
    axes[1,1].set_title('Orientation 90 Compare',size = font_size)
    axes[1,2].set_title('Orientation 135 Compare',size = font_size)
    axes[2,0].set_title('Red Map Compare',size = font_size)
    axes[2,1].set_title('Green Map Compare',size = font_size)
    axes[2,2].set_title('Blue Map Compare',size = font_size)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.03, hspace=None)
    fig.tight_layout()
    fig.figure.savefig(f'{axis_path}\\Recovered_Map_Compare_{cloc_name}.png')

ot.Save_Variable(work_path,'all_repeat_similar_frame',all_repeat_similar_frame)
ot.Save_Variable(work_path,'all_repeat_freq_frame',all_repeat_freq_frame)
ot.Save_Variable(work_path,'all_repeat_map_similar_frame',all_repeat_map_similar_frame)
ot.Save_Variable(work_path,'all_recovered_series',all_recovered_series)
#%% Stats on recovered map similarity
used_similarity = all_repeat_similar_frame[all_repeat_similar_frame['Frame_Count'] != 0]
used_similarity['Network'] = ''
for i in range(len(used_similarity)):
    c_id = used_similarity.iloc[i,1]
    if c_id<9:
        used_similarity.iloc[i,-1]='Eye'
    elif c_id<17 and c_id>8:
        used_similarity.iloc[i,-1]='Orientation'
    elif c_id>16:
        used_similarity.iloc[i,-1]='Color'
plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3,5),dpi = 180)
sns.boxplot(data = used_similarity,x = 'Network',y = 'Corr',showfliers = False,ax = ax,width = 0.35)
ax.set_title('Recovered Map Similarity')
#%% Stats on recovered map frequency.
plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5,5),dpi = 180)
sns.boxplot(data = all_repeat_freq_frame,x = 'Network',y = 'Freq',showfliers = False,ax = ax,width = 0.35)
ax.set(ylim = (0,0.1))
ax.set_title('Recovered Map Similarity')
#%% Payload analysis.
all_payload = pd.DataFrame(columns = ['Loc','Axes','Payload'])# just series.
all_payload_scatter = pd.DataFrame(columns = ['OD','HV','AO','From_Loc'])# scatter all points
for i,cc in tqdm(enumerate(all_path_dic)):
    cloc_name = cc.split('\\')[-1]
    c_PCA_results = all_PCA_results[cloc_name]
    c_OD_ax = c_PCA_results['OD_vec'][1:]
    c_HV_ax = c_PCA_results['HV_vec'][1:]
    c_AO_ax = c_PCA_results['AO_vec'][1:]
    c_OD_ax = c_OD_ax/np.linalg.norm(c_OD_ax)
    c_HV_ax = c_HV_ax/np.linalg.norm(c_HV_ax)
    c_AO_ax = c_AO_ax/np.linalg.norm(c_AO_ax)
    c_OD_payload = np.dot(c_PCA_results['Frame_Coords'][:,1:],c_OD_ax)
    c_HV_payload = np.dot(c_PCA_results['Frame_Coords'][:,1:],c_HV_ax)
    c_AO_payload = np.dot(c_PCA_results['Frame_Coords'][:,1:],c_AO_ax)
    all_payload.loc[len(all_payload),:] = [cloc_name,'OD',c_OD_payload]
    all_payload.loc[len(all_payload),:] = [cloc_name,'HV',c_HV_payload]
    all_payload.loc[len(all_payload),:] = [cloc_name,'AO',c_AO_payload]
    # and put scatters onto it.
    for j in range(len(c_OD_payload)):
        all_payload_scatter.loc[len(all_payload_scatter),:]=[c_OD_payload[j],c_HV_payload[j],c_AO_payload[j],cloc_name]
# used_frame = all_payload_scatter.groupby('From_Loc').get_group('L76_15A_220812')
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,10),dpi = 180)
pd.plotting.scatter_matrix(all_payload_scatter[['OD','HV','AO']].astype('f8'),ax = axes)
fig.suptitle('Network Load',fontsize = 20)
fig.tight_layout()
plt.show()
#%% stats of all data point without pooling.
all_point_scatters = list(all_payload_scatter.groupby('From_Loc'))
corr_frame = pd.DataFrame(columns= ['Loc','Network Pair','Corr'])
for i in range(len(all_point_scatters)):
    c_loc_frame = all_point_scatters[i][1]
    c_loc = all_point_scatters[i][0]
    od_hv_r,_ = stats.pearsonr(c_loc_frame['OD'],c_loc_frame['HV'])
    od_ao_r,_ = stats.pearsonr(c_loc_frame['OD'],c_loc_frame['AO'])
    hv_ao_r,_ = stats.pearsonr(c_loc_frame['HV'],c_loc_frame['AO'])
    corr_frame.loc[len(corr_frame),:] = [c_loc,'OD-HV',od_hv_r]
    corr_frame.loc[len(corr_frame),:] = [c_loc,'OD-AO',od_ao_r]
    corr_frame.loc[len(corr_frame),:] = [c_loc,'HV-AO',hv_ao_r]
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(2.5,4),dpi = 180)
sns.boxplot(data = corr_frame,x = 'Network Pair',y = 'Corr',width = 0.5)
fig.suptitle('Network Correlation')
fig.tight_layout()
plt.show()