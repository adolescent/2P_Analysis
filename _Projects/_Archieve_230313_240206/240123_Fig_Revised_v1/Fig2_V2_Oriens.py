'''
This code is almost the same as Single Netwoek UMAP, but we do this on V2.

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
from Cell_Class.UMAP_Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *



all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V2'))
all_path_dic.pop(4)
all_path_dic.pop(6)

work_path = r'D:\_Path_For_Figs\240123_Graph_Revised_v1\Fig2_Orien_Only_UMAP'
expt_folder = all_path_dic[2]
c_ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
c_ac.wp = expt_folder
c_spon = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
c_model = ot.Load_Variable(expt_folder,'Orien_UMAP_3D_20comp.pkl')
c_ac.Regenerate_Cell_Graph()

import warnings
warnings.filterwarnings("ignore")
#%% #############################STEP0, GENERATE EXAMPLE LOCS ####################

analyzer_orien = UMAP_Analyzer(ac = c_ac,umap_model = c_model,spon_frame = c_spon,orien = True,od = False,color = False)
analyzer_orien.Train_SVM_Classifier(C = 1)
g16_embeddings = analyzer_orien.stim_embeddings
g16_ids = analyzer_orien.stim_label
spon_embeddings = analyzer_orien.spon_embeddings
spon_ids = analyzer_orien.spon_label
# and add shuffle.
spon_frame_s = Spon_Shuffler(c_spon)
spon_embedding_s = c_model.transform(spon_frame_s)
spon_label_s = SVC_Fit(analyzer_orien.svm_classifier,data = spon_embedding_s,thres_prob = 0)

#%% ########################## STEP1, EXAMPLE EMBEDDINGS ################
import matplotlib.colors as mcolors
import matplotlib as mpl
import colorsys
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=3, ncols=1,figsize = (8,12),dpi = 180,subplot_kw=dict(projection='3d'))
## limit and title.
orien_elev = 15
orien_azim = -30
for i in range(3):
    ax[i].set_xlabel('UMAP 1')
    ax[i].set_ylabel('UMAP 2')
    ax[i].set_zlabel('UMAP 3')
    ax[i].grid(False)
    ax[i].view_init(elev=orien_elev, azim=orien_azim)
    ax[i].axes.set_xlim3d(left=-2, right=5) 
    ax[i].axes.set_ylim3d(bottom=-6, top=4) 
    ax[i].axes.set_zlim3d(bottom=3, top=10) 
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
rest_stim,_ = Select_Frame(g16_embeddings,g16_ids,used_id=[0])
orien_stim,orien_stim_id = Select_Frame(g16_embeddings,g16_ids,used_id=list(range(9,17)))
orien_color_stim = np.zeros(shape = (len(orien_stim_id),3),dtype='f8')
for i,c_id in enumerate(orien_stim_id):
    orien_color_stim[i,:] = color_setb[int(c_id)-9,:]
ax[0].scatter3D(rest_stim[:,0],rest_stim[:,1],rest_stim[:,2],s = 1,c = [0.7,0.7,0.7],alpha = 0.1)
ax[0].scatter3D(orien_stim[:,0],orien_stim[:,1],orien_stim[:,2],s = 1,c = orien_color_stim)
## colorize spons
rest_spon,_ = Select_Frame(spon_embeddings,spon_ids,used_id=[0])
orien_spon,orien_spon_id = Select_Frame(spon_embeddings,spon_ids,used_id=list(range(9,17)))
orien_color_spon = np.zeros(shape = (len(orien_spon_id),3),dtype='f8')
for i,c_id in enumerate(orien_spon_id):
    orien_color_spon[i,:] = color_setb[int(c_id)-9,:]
ax[1].scatter3D(rest_spon[:,0],rest_spon[:,1],rest_spon[:,2],s = 1,c = [0.7,0.7,0.7],alpha = 0.1)
ax[1].scatter3D(orien_spon[:,0],orien_spon[:,1],orien_spon[:,2],s = 1,c = orien_color_spon)
## colorize shuffles.
rest_shuffle,_ = Select_Frame(spon_embedding_s,spon_label_s,used_id=[0])
orien_shuffle,orien_shuffle_id = Select_Frame(spon_embedding_s,spon_label_s,used_id=list(range(9,17)))
orien_color_shuffle = np.zeros(shape = (len(orien_shuffle_id),3),dtype='f8')
for i,c_id in enumerate(orien_shuffle_id):
    orien_color_shuffle[i,:] = color_setb[int(c_id)-9,:]
ax[2].scatter3D(rest_shuffle[:,0],rest_shuffle[:,1],rest_shuffle[:,2],s = 1,c = [0.7,0.7,0.7],alpha = 0.1)
ax[2].scatter3D(orien_shuffle[:,0],orien_shuffle[:,1],orien_shuffle[:,2],s = 1,c = orien_color_shuffle)
# titles
ax[0].set_title('Stimulus Embedding in UMAP Space',size = 14)
ax[1].set_title('Spontaneous Embedding in UMAP Space',size = 14)
ax[2].set_title('Shuffled Embedding in UMAP Space',size = 14)

fig.tight_layout()


#%% ##########################2.RECOVER EXAMPLE MAPS##################
#Generate all repeat's correlation
orien_similar,orien_similar_rand = analyzer_orien.Similarity_Compare_All(id_lists=list(range(9,17)))
distribution_frame = pd.DataFrame(columns = ['Pearson R','Response Pattern','Data'])
for i,c_cond in enumerate(['Orientation']):
    c_data = [orien_similar][i]
    c_rand = [orien_similar_rand][i]
    for j,c_frame in enumerate(c_data):
        distribution_frame.loc[len(distribution_frame)] = [c_frame,c_cond,'Data']
    for j,c_frame in enumerate(c_rand):
        distribution_frame.loc[len(distribution_frame)] = [c_frame,c_cond,'Random']
# violin plot here.
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(2.5,4),dpi = 180)
axes.axhline(y = 0,color='gray', linestyle='--')
sns.violinplot(data=distribution_frame, x="Response Pattern", y="Pearson R",order = ['Orientation'],hue = 'Data',split=True, inner="quart",ax = axes,dodge= True,width=0.5)
axes.set_title('Spontaneous Repeat Similarity',size = 10)
axes.legend_.remove()
plt.tight_layout()
plt.show()
#%% Then show recovered map.
analyzer_orien.Similarity_Compare_Average(od = False,color = False)
value_max = 2.5
value_min = -1
font_size = 12
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10,5),dpi = 180)
cbar_ax = fig.add_axes([.99, .15, .02, .7])
sns.heatmap(analyzer_orien.stim_recover['Orien0'][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[0,0],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(analyzer_orien.spon_recover['Orien0'][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[1,0],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(analyzer_orien.stim_recover['Orien45'][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[0,1],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(analyzer_orien.spon_recover['Orien45'][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[1,1],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(analyzer_orien.stim_recover['Orien90'][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[0,2],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(analyzer_orien.spon_recover['Orien90'][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[1,2],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(analyzer_orien.stim_recover['Orien135'][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[0,3],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(analyzer_orien.spon_recover['Orien135'][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[1,3],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)


axes[0,0].set_title('Orientation0',size = font_size)
axes[0,1].set_title('Orientation45',size = font_size)
axes[0,2].set_title('Orientation90',size = font_size)
axes[0,3].set_title('Orientation135',size = font_size)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=None)
fig.tight_layout()
plt.show()
#%%######################3. STATS OF ALL V2###################################
# 1. Generate all repeat with similarity.
for i,c_loc in tqdm(enumerate(all_path_dic)):
    c_loc_last = c_loc.split('\\')[-1]
    c_spon_frame = ot.Load_Variable(c_loc,'Spon_Before.pkl')
    c_ac = ot.Load_Variable_v2(c_loc,'Cell_Class.pkl')
    model_orien = ot.Load_Variable(c_loc,'Orien_UMAP_3D_20comp.pkl')
    # trian reducer with only G16 data.
    g16_frames,g16_ids = c_ac.Combine_Frame_Labels(od = False,color = False)
    
    c_analyzer = UMAP_Analyzer(ac = c_ac,umap_model=model_orien,spon_frame=c_spon_frame,od = False,color = False)
    c_analyzer.Train_SVM_Classifier(C = 1)
    c_analyzer.Similarity_Compare_Average(od = False,color = False)
    c_recover_similar = c_analyzer.Avr_Similarity
    c_recover_similar['Location'] = c_loc_last
    if i == 0:
        all_recover_similarity = copy.deepcopy(c_recover_similar)
    else:
        all_recover_similarity = pd.concat([all_recover_similarity,c_recover_similar],ignore_index=True)
#%% 2.Plot averaged maps' similarity
plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (2.5,5),dpi = 180)
ax.axhline(y = 0,color='gray', linestyle='--')
sns.boxplot(data = all_recover_similarity,x = 'Data',y = 'PearsonR',hue = 'Network',ax = ax,showfliers = False)
ax.set_title('Network Repeat Similarity')
ax.set_xlabel('')
ax.legend(title = 'Network',fontsize = 8)
ax.set_xticklabels(['Real Data','Random Select'],size = 7)
ax.set_ylabel('Pearson R')
plt.show()
#%% ########################## ALL REPEAT FREQUENCY STATS ###############################
spon_repeat_count = pd.DataFrame(columns=['Loc','Network','Repeat_Freq','Repeat_Num','Data'])
for i,c_loc in tqdm(enumerate(all_path_dic)):
    c_loc_last = c_loc.split('\\')[-1]
    c_spon_frame = ot.Load_Variable(c_loc,'Spon_Before.pkl')
    c_ac = ot.Load_Variable_v2(c_loc,'Cell_Class.pkl')
    g16_frames,g16_ids = c_ac.Combine_Frame_Labels(od = False,color = False)
    model_orien = ot.Load_Variable(c_loc,'Orien_UMAP_3D_20comp.pkl')
    c_analyzer = UMAP_Analyzer(ac = c_ac,umap_model=model_orien,spon_frame=c_spon_frame,od= False,color= False)
    c_analyzer.Train_SVM_Classifier(C = 1)
    c_analyzer.Similarity_Compare_Average()
    c_spon_series = c_analyzer.spon_label
    tc_analyzer = Series_TC_info(input_series=c_spon_series,od = False,color= False)
    _,c_orien_freq,_ = tc_analyzer.Freq_Estimation(type='Event')
    _,c_orien_num,_ = tc_analyzer.Freq_Estimation(type='Frame')
    # and shuffle network repeat freq.
    shuffle_times = 10
    repeat_freq_s = np.zeros(shape = (shuffle_times,1),dtype = 'f8')# save in sequence od,orien,color.
    for j in range(shuffle_times):# shuffle
        spon_frame_s = Spon_Shuffler(c_spon_frame,method='phase')
        spon_embedding_s = model_orien.transform(spon_frame_s)
        c_spon_label_s = SVC_Fit(c_analyzer.svm_classifier,data = spon_embedding_s,thres_prob = 0)
        tc_analyzer_s = Series_TC_info(input_series=c_spon_label_s,od = False,color = False)
        _,c_orien_freq_s,_ = tc_analyzer_s.Freq_Estimation(type='Event')
        _,c_orien_num_s,_ = tc_analyzer_s.Freq_Estimation(type='Frame')
        repeat_freq_s[j,:] = [c_orien_freq_s]
    repeat_freq_s = repeat_freq_s.mean(0)
    spon_repeat_count.loc[len(spon_repeat_count),:] = [c_loc_last,'Orientation',c_orien_freq,c_orien_num,'Real']
    spon_repeat_count.loc[len(spon_repeat_count),:] = [c_loc_last,'Orientation',c_orien_freq_s,c_orien_num_s,'Shuffle']
#%% visualization
plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (2.5,5),dpi = 180)
ax.axhline(y = 0,color='gray', linestyle='--')
sns.boxplot(data = spon_repeat_count,hue = 'Data',y = 'Repeat_Freq',x = 'Network',ax = ax,showfliers = False,width = 0.5)
ax.set_title('Stim-like Ensemble Repeat Frequency',size = 10)
ax.set_xlabel('')
ax.set_ylabel('Repeat Frequency(Hz)')
ax.legend(title = 'Network',fontsize = 8)
# ax.set_xticklabels(['Real Data','Random Select'],size = 7)
plt.show()
