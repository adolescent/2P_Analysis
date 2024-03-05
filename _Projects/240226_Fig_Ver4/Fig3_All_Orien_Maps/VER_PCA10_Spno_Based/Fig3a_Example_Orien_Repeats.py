'''
This file shows an example of orien corr's cos similarity.
Loc L76-18M Used.

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

work_path = r'D:\_Path_For_Figs\240228_Figs_v4\Fig3'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
# some times we need to ignore warnings.
import warnings
warnings.filterwarnings("ignore")

example_loc = r'D:\_All_Spon_Data_V1\L76_18M_220902'
all_orien_maps = ot.Load_Variable(work_path,'VAR1_All_Orien_Response.pkl')
all_cell_oriens = ot.Load_Variable(work_path,'VAR2_All_Cell_Best_Oriens.pkl')

#%% ###################### 0.Define Basic Functions.############################
def Find_Example(corr_mat,spon_label,c_spon,center = 30,width = 5,min_corr =0.2):
    c_spon = np.array(c_spon)
    find_from = corr_mat[corr_mat.min(1)<min_corr]
    best_locs = find_from.idxmax(1)
    satistied_series = np.where((best_locs>(center-width))*(best_locs<(center+width)))[0]
    # best_id = Corr_Matrix_Norm.loc[satistied_series,:].max(1).idxmax()
    best_id = find_from.iloc[satistied_series,:].max(1).idxmax()
    origin_class = spon_label[best_id]
    origin_frame = ac.Generate_Weighted_Cell(c_spon[best_id,:])
    corr_series = corr_mat.loc[best_id,:]
    best_orien = corr_series.idxmax()
    best_corr = corr_series.max()
    print(f'Best Orientation {best_orien}, with corr {best_corr}.')
    print(f'PCA Classified Class:{origin_class}')
    return origin_frame,origin_class,corr_series,best_orien,best_corr,best_id

#%% ###################### 1. Get All Orien Corrs of Example Locs.#################################
#%% get basic spontaneous embeddings and infos.
pc_num = 10
ac = ot.Load_Variable_v2(example_loc,'Cell_Class.pkl')
c_spon = np.array(ot.Load_Variable_v2(example_loc,'Spon_Before.pkl'))
spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=c_spon,sample='Frame',pcnum=pc_num)
# c_oriens_stimon,c_oriens_stimon_ids = ac.Combine_Frame_Labels(od = 0,orien = 1,color = 0,isi = 0)
analyzer = UMAP_Analyzer(ac = ac,umap_model=spon_models,spon_frame=c_spon,od = 0,orien = 1,color = 0,isi = True)
analyzer.Train_SVM_Classifier(C=1)
stim_embed = analyzer.stim_embeddings
stim_label = analyzer.stim_label
spon_embed = analyzer.spon_embeddings
spon_label = analyzer.spon_label
cloc_spon_patterns = all_orien_maps['L76_18M_220902']
# get corr mat of current spon locs.
# predicted_spon = c_spon[spon_label>0,:]
predicted_spon = c_spon
Corr_Matrix = pd.DataFrame(0.0,columns = cloc_spon_patterns.columns,index = range(len(predicted_spon)))
for i in tqdm(range(len(predicted_spon))):
    single_spon = predicted_spon[i,:]
    for j in range(cloc_spon_patterns.shape[1]):
        c_pattern = np.array(cloc_spon_patterns.iloc[:,j])
        cos_sim = single_spon.dot(c_pattern) / (np.linalg.norm(single_spon) * np.linalg.norm(c_pattern))
        Corr_Matrix.iloc[i,j] = cos_sim

#%% ##########################  FIG 3A - 4 EXAMPLE ################################
# 1. Find 4 examples and calculate example corrs.
c_corr_frames = Corr_Matrix
min_corr = 0.2
temp1,temp1_class,temp1_corr,temp1_best_orien,temp1_best_corr,temp1_id = Find_Example(c_corr_frames,spon_label,c_spon,13,min_corr = min_corr )
temp2,temp2_class,temp2_corr,temp2_best_orien,temp2_best_corr,temp2_id = Find_Example(c_corr_frames,spon_label,c_spon,55,min_corr = min_corr )
temp3,temp3_class,temp3_corr,temp3_best_orien,temp3_best_corr,temp3_id = Find_Example(c_corr_frames,spon_label,c_spon,102,min_corr = min_corr )
temp4,temp4_class,temp4_corr,temp4_best_orien,temp4_best_corr,temp4_id = Find_Example(c_corr_frames,spon_label,c_spon,155,min_corr = min_corr )
# and get 2 non - repeat frames.
non1 = 5
non2 = 5349
temp5 = ac.Generate_Weighted_Cell(c_spon[non1,:])
temp5_corr = np.array(c_corr_frames.iloc[non1,:])
temp6 = ac.Generate_Weighted_Cell(c_spon[non2,:])
temp6_corr = np.array(c_corr_frames.iloc[non2,:])

plt.plot(temp1_corr)
plt.plot(temp2_corr)
plt.plot(temp3_corr)
plt.plot(temp4_corr)
plt.plot(temp5_corr)
plt.plot(temp6_corr)

#%% and Plot all 6 examples with radar map.
value_max = 5
value_min = -3
r_min = -0.3
r_max = 1
# r_ticks = [-0.3,0,0.3,0.6,0.9]
r_ticks = [0,0.5]
radius_zoom = 0.5
plt.clf()
plt.cla()
# fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(12,3),dpi = 180)
fig = plt.figure(figsize=(8,9),dpi = 180)
axes = []

for i in range(12):
    if i%2 ==0:
        axes.append(plt.subplot(3,4,i+1))
    else:
        axes.append(plt.subplot(3,4,i+1, projection='polar'))
cbar_ax = fig.add_axes([.97, .4, .03, .3])
# all example frames
all_examples = [temp1,temp2,temp3,temp4,temp5,temp6]
for i,c_graph in enumerate(all_examples):
    sns.heatmap(c_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[2*i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    axes[2*i].set_title(f'Example Frame {i+1}',size = 10)

all_corrs = [temp1_corr,temp2_corr,temp3_corr,temp4_corr,temp5_corr,temp6_corr]
fig.subplots_adjust(left=None, bottom=None, right=0.99, top=None, wspace=0.02, hspace=None)

for i,c_corr in enumerate(all_corrs):
    # zoom in graph
    axes[2*i+1].set_position([axes[2*i+1].get_position().x0+0.05, axes[2*i+1].get_position().y0+0.05, axes[2*i+1].get_position().width*radius_zoom, axes[2*i+1].get_position().height*radius_zoom])
    # plot graphs 
    axes[2*i+1].plot(np.linspace(0,2*np.pi,180), c_corr) 
    axes[2*i+1].plot(np.linspace(0,2*np.pi,180), np.zeros(180),color = 'black',linewidth = 0.7,linestyle = '-') 
    axes[2*i+1].plot(np.linspace(0,2*np.pi,180), np.zeros(180)+0.5,color = 'gray',linewidth = 0.7,linestyle = '-') 
    axes[2*i+1].set_rlim(r_min,r_max)
    axes[2*i+1].set_xticks(np.arange(0, 2*np.pi, 2*np.pi/6))
    axes[2*i+1].set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°'],size = 6)
    axes[2*i+1].set_rticks(r_ticks)
    axes[2*i+1].set_yticklabels(r_ticks,fontsize=6)
    axes[2*i+1].set_rlabel_position(45) 
    # axes[2*i+1].set_title('Cosine Similarity',size = 9)
    # axes[2*i+1].set_xlabel('Angle')
# fig.tight_layout()
plt.show()



#%% ##########################  FIG 3A-2 - ALL COSINE SIMILARITY ################################
# get normalized graphs
c_corr_frames_normed = copy.deepcopy(c_corr_frames)
for i in range(c_corr_frames.shape[1]):
    c_column_std = c_corr_frames.iloc[:,i].std()
    c_corr_frames_normed.iloc[:,i] = c_corr_frames_normed.iloc[:,i]/c_column_std

# on_parts = c_corr_frames_normed.iloc[spon_label>0,:]
# off_parts = c_corr_frames_normed.iloc[spon_label==0,:]
on_parts = c_corr_frames.iloc[spon_label>0,:]
off_parts = c_corr_frames.iloc[spon_label==0,:]
on_parts['Best_Angle'] = on_parts.idxmax(1)
on_parts_sorted = on_parts.sort_values(by=['Best_Angle'])
on_parts_sorted  = on_parts_sorted.drop(['Best_Angle'],axis = 1)
on_parts  = on_parts.drop(['Best_Angle'],axis = 1)
off_parts['Best_Angle'] = off_parts.idxmax(1)
off_parts_sorted = off_parts.sort_values(by=['Best_Angle'])
off_parts_sorted  = off_parts_sorted.drop(['Best_Angle'],axis = 1)
off_parts  = off_parts.drop(['Best_Angle'],axis = 1)

sorted_mat = pd.concat([on_parts_sorted,off_parts_sorted])
plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4),dpi = 180)
sns.heatmap(sorted_mat.iloc[:,:-1],center = 0,vmax = 1,vmin = -1,xticklabels=False,yticklabels=False,ax = ax)
# Corr_Matrix_Norm = Corr_Matrix_Norm.drop(['Best_Angle'],axis = 1)
ax.set_title('Similarity with All Orientation Maps')
ax.set_ylabel('Frames')
ax.set_xticks([0,45,90,135])
ax.set_xticklabels([0,45,90,135])
ax.set_xlabel('Orientation Angles')
sns.lineplot(x=[0,45,90,135,180], y=len(on_parts),color = 'y')
fig.tight_layout()
plt.show()

#%% ##########################  FIG 3A-1-Ver2 Frame Repeats ################################
# get single frame repeat of this graph.
raw_spon_frames = ot.Load_Variable(example_loc,'Spon_Before_Raw.pkl')
spon_avr = raw_spon_frames.mean(0)

temp1_raw = raw_spon_frames[temp1_id,:,:]-spon_avr
temp2_raw = raw_spon_frames[temp2_id,:,:]-spon_avr
temp3_raw = raw_spon_frames[temp3_id,:,:]-spon_avr
temp4_raw = raw_spon_frames[temp4_id,:,:]-spon_avr
temp5_raw = raw_spon_frames[5,:,:]-spon_avr
temp6_raw = raw_spon_frames[5349,:,:]-spon_avr

#%% Plot frame graphs
# value_max = 4
# value_min = -3.5
value_max = 1500
value_min = -1500
clip_std = 5

example_frames = np.zeros(shape = (6,512,512),dtype = 'f8')
for i,c_graph in enumerate([temp1_raw,temp2_raw,temp3_raw,temp4_raw,temp5_raw,temp6_raw]):
    c_graph_clip = np.clip(c_graph,c_graph.mean()-c_graph.std()*clip_std,c_graph.mean()+c_graph.std()*clip_std)
    example_frames[i,:,:] = c_graph_clip


plt.clf()
plt.cla()
fig,axes = plt.subplots(nrows=3, ncols=2,figsize=(6,9),dpi = 180)
cbar_ax = fig.add_axes([.99, .3, .03, .4])

for i in range(6):
    plotable_frame = example_frames[i,:,:]
    # plotable_frame = plotable_frame/plotable_frame.std()
    sns.heatmap(plotable_frame,center = 0,xticklabels=False,yticklabels=False,ax = axes[i//2,i%2],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    axes[i//2,i%2].set_title(f'Example Frame {i+1}')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
fig.tight_layout()
plt.show()

