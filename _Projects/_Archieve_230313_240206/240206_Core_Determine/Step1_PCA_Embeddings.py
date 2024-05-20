'''
This script will generate PCA analysis on G16 data, aim at comparing this method with umap methods.

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
from Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *


# work_path = r'D:\_Path_For_Figs\240123_Graph_Revised_v1\Fig1_Revised'
expt_folder = r'D:\_All_Spon_Data_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
ac.wp = expt_folder
c_spon = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
c_model = ot.Load_Variable(expt_folder,'All_Stim_UMAP_3D_20comp.pkl')
ac.Regenerate_Cell_Graph()

work_path = r'D:\_Path_For_Figs\240221_PCA_SVM'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
import warnings
warnings.filterwarnings("ignore")

#%% ################################ 1, DO PCA ON G16 DATAS. ######################
# 1. fit G16 models
ac = ot.Load_Variable_v2(all_path_dic[2],'Cell_Class.pkl')
c_spon = ot.Load_Variable(all_path_dic[2],'Spon_Before.pkl')
pcnum = 10
g16_frames,g16_labels = ac.Combine_Frame_Labels(od = 0,color = 0,orien = 1)
# g16_frames,g16_labels = ac.Combine_Frame_Labels(od = False,color = True,orien = False)
g16_pcs,g16_coords,g16_models = Z_PCA(Z_frame=g16_frames,sample='Frame',pcnum=pcnum)

model_var_ratio = np.array(g16_models.explained_variance_ratio_)
print(f'{pcnum} PCs explain G16 VAR {model_var_ratio[:pcnum].sum()*100:.1f}%')
# 2. Use Analyzer to generate spon labels 
analyzer = Classify_Analyzer(ac = ac,umap_model=g16_models,spon_frame=c_spon,od = 0, color = 0,orien = 1)
analyzer.Train_SVM_Classifier()
spon_label = analyzer.spon_label
# 3. try phase shuffle and dim shuffle. 
spon_s_phase = Spon_Shuffler(c_spon,method='phase')
spon_s_dim = Spon_Shuffler(c_spon,method='dim')
# get labels of phase shuffled and dim shuffled frames.
spon_s_phase_embeddings = g16_models.transform(spon_s_phase)
spon_s_phase_label = SVC_Fit(analyzer.svm_classifier,spon_s_phase_embeddings,0)
spon_s_dim_embeddings = g16_models.transform(spon_s_dim)
spon_s_dim_label = SVC_Fit(analyzer.svm_classifier,spon_s_dim_embeddings,0)

print((spon_label>0).sum())
print((spon_s_phase_label>0).sum())
print((spon_s_dim_label>0).sum())
#%%##################################### 2, GET EMBEDDING G16 GRAPHS ####################################

import matplotlib.colors as mcolors
import matplotlib as mpl
import colorsys
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=4, ncols=1,figsize = (8,16),dpi = 180,subplot_kw=dict(projection='3d'))
## limit and title.
used_dims = [1,2,3]
# used_dims = [0,1,2]
orien_elev = 25
orien_azim = 60
for i in range(4):
    ax[i].set_xlabel('PC 2')
    ax[i].set_ylabel('PC 3')
    ax[i].set_zlabel('PC 4')
    ax[i].grid(False)
    ax[i].view_init(elev=orien_elev, azim=orien_azim)
    # ax[i].axes.set_xlim3d(left=-40, right=60)
    # ax[i].axes.set_ylim3d(bottom=-50, top=50)
    # ax[i].axes.set_zlim3d(bottom=-30, top=30)
    ax[i].axes.set_xlim3d(left=-20, right=20)
    ax[i].axes.set_ylim3d(bottom=-20, top=20)
    ax[i].axes.set_zlim3d(bottom=-20, top=20)
## get orien color bars.
color_setb = np.zeros(shape = (8,3))
for i,c_orien in enumerate(np.arange(0,180,22.5)):
    c_hue = c_orien/180
    c_lightness = 0.5
    c_saturation = 1
    color_setb[i,:] = colorsys.hls_to_rgb(c_hue, c_lightness, c_saturation)
cax_b = fig.add_axes([0.15, 0.4, 0.02, 0.3])
custom_cmap = mcolors.ListedColormap(color_setb)
bounds = np.arange(0,202.5,22.5)
norm = mpl.colors.BoundaryNorm(bounds, custom_cmap.N)
c_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=custom_cmap),cax=cax_b, label='Orientation')
c_bar.set_ticks(np.arange(0,180,22.5)+11.25)
c_bar.set_ticklabels(np.arange(0,180,22.5))
c_bar.ax.tick_params(size=0)

## colorize stims
g16_embeddings = g16_coords[:,used_dims]
g16_ids = analyzer.stim_label
rest_stim,_ = Select_Frame(g16_embeddings,g16_ids,used_id=[0])
orien_stim,orien_stim_id = Select_Frame(g16_embeddings,g16_ids,used_id=list(range(9,17)))
orien_color_stim = np.zeros(shape = (len(orien_stim_id),3),dtype='f8')
for i,c_id in enumerate(orien_stim_id):
    orien_color_stim[i,:] = color_setb[int(c_id)-9,:]
ax[0].scatter3D(rest_stim[:,0],rest_stim[:,1],rest_stim[:,2],s = 1,c = [0.7,0.7,0.7],alpha = 0.1)
ax[0].scatter3D(orien_stim[:,0],orien_stim[:,1],orien_stim[:,2],s = 1,c = orien_color_stim)

## colorize spons
spon_embeddings = analyzer.spon_embeddings[:,used_dims]
spon_ids = analyzer.spon_label
rest_spon,_ = Select_Frame(spon_embeddings,spon_ids,used_id=[0])
orien_spon,orien_spon_id = Select_Frame(spon_embeddings,spon_ids,used_id=list(range(9,17)))
orien_color_spon = np.zeros(shape = (len(orien_spon_id),3),dtype='f8')
for i,c_id in enumerate(orien_spon_id):
    orien_color_spon[i,:] = color_setb[int(c_id)-9,:]
ax[1].scatter3D(rest_spon[:,0],rest_spon[:,1],rest_spon[:,2],s = 1,c = [0.7,0.7,0.7],alpha = 0.1)
ax[1].scatter3D(orien_spon[:,0],orien_spon[:,1],orien_spon[:,2],s = 1,c = orien_color_spon)

## colorize shuffle phase.
spon_embeddings_s = spon_s_phase_embeddings[:,used_dims]
spon_ids_s = spon_s_phase_label
rest_shuffle,_ = Select_Frame(spon_embeddings_s,spon_ids_s,used_id=[0])
orien_shuffle,orien_shuffle_id = Select_Frame(spon_embeddings_s,spon_ids_s,used_id=list(range(9,17)))
orien_color_shuffle = np.zeros(shape = (len(orien_shuffle_id),3),dtype='f8')
for i,c_id in enumerate(orien_shuffle_id):
    orien_color_shuffle[i,:] = color_setb[int(c_id)-9,:]
ax[2].scatter3D(rest_shuffle[:,0],rest_shuffle[:,1],rest_shuffle[:,2],s = 1,c = [0.7,0.7,0.7],alpha = 0.1)
ax[2].scatter3D(orien_shuffle[:,0],orien_shuffle[:,1],orien_shuffle[:,2],s = 1,c = orien_color_shuffle)

## colorize shuffle dims.
spon_embeddings_s = spon_s_dim_embeddings[:,used_dims]
spon_ids_s = spon_s_dim_label
rest_shuffle,_ = Select_Frame(spon_embeddings_s,spon_ids_s,used_id=[0])
orien_shuffle,orien_shuffle_id = Select_Frame(spon_embeddings_s,spon_ids_s,used_id=list(range(9,17)))
orien_color_shuffle = np.zeros(shape = (len(orien_shuffle_id),3),dtype='f8')
for i,c_id in enumerate(orien_shuffle_id):
    orien_color_shuffle[i,:] = color_setb[int(c_id)-9,:]
ax[3].scatter3D(rest_shuffle[:,0],rest_shuffle[:,1],rest_shuffle[:,2],s = 1,c = [0.7,0.7,0.7],alpha = 0.1)
ax[3].scatter3D(orien_shuffle[:,0],orien_shuffle[:,1],orien_shuffle[:,2],s = 1,c = orien_color_shuffle)

# set title
ax[0].set_title('Stimulus Embedding in PCA Space',size = 10)
ax[1].set_title('Spontaneous Embedding in PCA Space',size = 10)
ax[2].set_title('Shuffled Phase Embedding in PCA Space',size = 10)
ax[3].set_title('Shuffled Dim Embedding in PCA Space',size = 10)
for i in range(4):
        ax[i].set_box_aspect(aspect=None, zoom=0.9)

fig.tight_layout()

#%%###################################### 3, GET RECOVERED ORIEN GRAPHS ############################
analyzer.Get_Stim_Spon_Compare(od = False,color = False)

stim_graphs = analyzer.stim_recover
spon_graphs = analyzer.spon_recover
graph_lists = ['Orien0','Orien45','Orien90','Orien135']

plt.clf()
plt.cla()
value_max = 2
value_min = -1
font_size = 16
fig,axes = plt.subplots(nrows=2, ncols=4,figsize = (14,6),dpi = 180)
cbar_ax = fig.add_axes([.99, .15, .02, .7])

for i,c_map in enumerate(graph_lists):
    sns.heatmap(stim_graphs[c_map][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[0,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    sns.heatmap(spon_graphs[c_map][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[1,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    axes[0,i].set_title(c_map,size = font_size)

axes[0,0].set_ylabel('Stimulus',rotation=90,size = font_size)
axes[1,0].set_ylabel('Spontaneous',rotation=90,size = font_size)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
fig.tight_layout()
plt.show()

#%%#################################### 4, STATS OF ALL SIMILARITY###################################
# calculate all 
analyzer.Similarity_Compare_Average(od = False,color = False)
all_corr = analyzer.Avr_Similarity

#%%###################################5, STATS OF ALL RESPONSE FREQS.#################################

pcnum = 10
all_g16_explained_VAR = []
N_shuffle = 10
all_repeat_similarity = pd.DataFrame(columns = ['Loc','Network','Corr','Map_Type','Data_Type'])
all_repeat_freq = pd.DataFrame(columns = ['Loc','Network','Freq','Frame_Prop','Data_Type'])


for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    # g16_frames,g16_labels = ac.Combine_Frame_Labels(od = 0,color = 0,orien = 1)
    g16_frames,g16_labels = ac.Combine_Frame_Labels(od = 0,color = 0,orien = 1)
    g16_pcs,g16_coords,g16_models = Z_PCA(Z_frame=g16_frames,sample='Frame',pcnum=pcnum)
    model_var_ratio = np.array(g16_models.explained_variance_ratio_)
    all_g16_explained_VAR.append(model_var_ratio[:pcnum].sum())
    print(f'{pcnum} PCs explain G16 VAR {model_var_ratio[:pcnum].sum()*100:.1f}%')
    # analyzer = UMAP_Analyzer(ac = ac,umap_model=g16_models,spon_frame=c_spon,od = 0, color = 0,orien = 1)
    analyzer = Classify_Analyzer(ac = ac,umap_model=g16_models,spon_frame=c_spon,od = 0, color = 0,orien = 1)
    analyzer.Train_SVM_Classifier()
    analyzer.Similarity_Compare_Average(od = 0,color = 0,orien = 1)
    all_orien_corrs = analyzer.Avr_Similarity
    for j in range(len(all_orien_corrs)):
        c_map_info = all_orien_corrs.iloc[j,:]
        all_repeat_similarity.loc[len(all_repeat_similarity),:] = [cloc_name,c_map_info['Network'],c_map_info['PearsonR'],c_map_info['MapType'],c_map_info['Data']]

    spon_label = analyzer.spon_label
    g16_frames = (spon_label>0).sum()/len(spon_label)
    g16_events = Event_Counter(spon_label>0)*1.301/len(spon_label)
    all_repeat_freq.loc[len(all_repeat_freq),:] = [cloc_name,'Orien',g16_events,g16_frames,'Real_Data']
    for j in tqdm(range(N_shuffle)):
        spon_s_phase = Spon_Shuffler(c_spon,method='phase')
        spon_s_dim = Spon_Shuffler(c_spon,method='dim')
        # get labels of phase shuffled and dim shuffled frames.
        spon_s_phase_embeddings = g16_models.transform(spon_s_phase)
        spon_s_phase_label = SVC_Fit(analyzer.svm_classifier,spon_s_phase_embeddings,0)
        spon_s_dim_embeddings = g16_models.transform(spon_s_dim)
        spon_s_dim_label = SVC_Fit(analyzer.svm_classifier,spon_s_dim_embeddings,0)
        g16_frames_phase_s = (spon_s_phase_label>0).sum()/len(spon_label)
        g16_events_phase_s = Event_Counter(spon_s_phase_label>0)*1.301/len(spon_label)
        g16_frames_dim_s = (spon_s_dim_label>0).sum()/len(spon_label)
        g16_events_dim_s = Event_Counter(spon_s_dim_label>0)*1.301/len(spon_label)
        # save shuffle 
        all_repeat_freq.loc[len(all_repeat_freq),:] = [cloc_name,'Orien',g16_frames_phase_s,g16_events_phase_s,'Phase_Shuffle']
        all_repeat_freq.loc[len(all_repeat_freq),:] = [cloc_name,'Orien',g16_frames_dim_s,g16_events_dim_s,'Dim_Shuffle']

# ot.Save_Variable(work_path,'V5_All_Hue_Repeat_Similarity',all_repeat_similarity)
# ot.Save_Variable(work_path,'V6_All_Hue_Repeat_Frequency',all_repeat_freq)
#%% Plot repeat similarity
plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (2,5),dpi = 180)
ax.axhline(y = 0,color='gray', linestyle='--')
sns.boxplot(data = all_repeat_similarity,x = 'Data_Type',y = 'Corr',hue = 'Network',ax = ax,showfliers = False)
ax.set_title('Stim-like Ensemble Repeat Similarity',size = 10)
ax.set_xlabel('')
ax.set_ylabel('Pearson R')
ax.legend(title = 'Network',fontsize = 8)
ax.set_xticklabels(['Real Data','Random Select'],size = 7)
plt.show()
#%% And plot frequency
plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (7,5),dpi = 180)
ax.axhline(y = 0,color='gray', linestyle='--')

sns.boxplot(data = all_repeat_freq,x = 'Loc',y = 'Freq',hue = 'Data_Type',ax = ax,showfliers = 0,legend = True)
ax.set_title('Stim-like Ensemble Repeat Similarity',size = 10)
ax.set_xlabel('')
ax.set_ylabel('Frequency(Hz)')
ax.legend(title = 'Network',fontsize = 8)
# ax.set_xticklabels(['Real Data','Random Select'],size = 7)
ax.set_xticklabels([''],size = 7)
plt.show()

