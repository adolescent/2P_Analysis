'''
This script will redo fig3, use G16 only model and compare repeat-nonrepeat, and plot radar map instead of linear corr.

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


work_path = r'D:\_Path_For_Figs\240123_Graph_Revised_v1\Fig3_180Oriens_New'
expt_folder = r'D:\_All_Spon_Data_V1\L76_18M_220902'

all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)

import warnings
warnings.filterwarnings("ignore")
# Load in useful results.
all_spon_corr_dic = ot.Load_Variable(r'D:\_Path_For_Figs\2401_Amendments\Fig3_New\All_Location_Corr_Matrix.pkl')
#%% ######################0. Vital Functions#####################
def Find_Example(corr_mat,spon_label,c_spon,center = 30,width = 3,min_corr =0.5):
    find_from = corr_mat[corr_mat.min(1)<min_corr]
    best_locs = find_from.idxmax(1)
    satistied_series = np.where((best_locs>(center-width))*(best_locs<(center+width)))[0]
    # best_id = Corr_Matrix_Norm.loc[satistied_series,:].max(1).idxmax()
    best_id = find_from.iloc[satistied_series,:].max(1).idxmax()
    origin_class = spon_label[best_id]
    origin_frame = ac.Generate_Weighted_Cell(c_spon.iloc[best_id,:])
    corr_series = corr_mat.loc[best_id,:]
    best_orien = corr_series.idxmax()
    best_corr = corr_series.max()
    print(f'Best Orientation {best_orien}, with corr {best_corr}.')
    print(f'UMAP Classified Class:{origin_class}')
    return origin_frame,origin_class,corr_series,best_orien,best_corr


#%% ####################### 1.GET EXAMPLE CORRS #######################
# useful vars
cloc = all_path_dic[2]
ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
c_spon = ot.Load_Variable_v2(cloc,'Spon_Before.pkl')
c_model = ot.Load_Variable(cloc,'Orien_UMAP_3D_20comp.pkl')
analyzer = UMAP_Analyzer(ac = ac,umap_model=c_model,spon_frame=c_spon,od = False,color = False)
analyzer.Train_SVM_Classifier(C = 1)
spon_labels = analyzer.spon_label
stim_labels = analyzer.stim_label
c_corr_frames = all_spon_corr_dic[cloc.split('\\')[-1]]
on_parts = c_corr_frames.iloc[spon_labels>0,:]
off_parts = c_corr_frames.iloc[spon_labels==0,:]
# get 4 examples 
temp1,temp1_class,temp1_corr,temp1_best_orien,temp1_best_corr = Find_Example(c_corr_frames,spon_labels,c_spon,13)
temp2,temp2_class,temp2_corr,temp2_best_orien,temp2_best_corr = Find_Example(c_corr_frames,spon_labels,c_spon,55)
temp3,temp3_class,temp3_corr,temp3_best_orien,temp3_best_corr = Find_Example(c_corr_frames,spon_labels,c_spon,102)
temp4,temp4_class,temp4_corr,temp4_best_orien,temp4_best_corr = Find_Example(c_corr_frames,spon_labels,c_spon,150)
# and 2 non repeat examples.
non1 = 5
non2 = 5349
temp5 = ac.Generate_Weighted_Cell(c_spon.iloc[non1,:])
temp5_corr = np.array(c_corr_frames.iloc[non1,:])
temp6 = ac.Generate_Weighted_Cell(c_spon.iloc[non2,:])
temp6_corr = np.array(c_corr_frames.iloc[non2,:])
#%% Plot parts.
value_max = 5
value_min = -3
r_min = -1
r_max = 2
r_ticks = [0,1,2]
plt.clf()
plt.cla()
# fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(12,3),dpi = 180)
fig = plt.figure(figsize=(8,7),dpi = 180)
axes = []
for i in range(12):
    if i%2 ==0:
        axes.append(plt.subplot(3,4,i+1))
    else:
        axes.append(plt.subplot(3,4,i+1, projection='polar'))
cbar_ax = fig.add_axes([.95, .4, .03, .3])
# all example frames
all_examples = [temp1,temp2,temp3,temp4,temp5,temp6]
for i,c_graph in enumerate(all_examples):
    sns.heatmap(c_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[2*i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    axes[2*i].set_title(f'Example Frame {i+1}',size = 10)
# and polar corrs.
# axes[0,1] = plt.subplot(262, projection='polar')
all_corrs = [temp1_corr,temp2_corr,temp3_corr,temp4_corr,temp5_corr,temp6_corr]
for i,c_corr in enumerate(all_corrs):
    # zoom in graph
    axes[2*i+1].set_position([axes[2*i+1].get_position().x0+0.02, axes[2*i+1].get_position().y0+0.02, axes[2*i+1].get_position().width*0.75, axes[2*i+1].get_position().height*0.75])
    # plot graphs 
    axes[2*i+1].plot(np.linspace(0,2*np.pi,180), c_corr) 
    # axes[1].plot(np.linspace(0,2*np.pi,180), np.zeros(180),color = 'gray',linestyle = '--') 
    axes[2*i+1].set_rlim(r_min,r_max)
    axes[2*i+1].set_xticks(np.arange(0, 2*np.pi, 2*np.pi/6))
    axes[2*i+1].set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°'])
    axes[2*i+1].set_rticks(r_ticks)
    axes[2*i+1].set_rlabel_position(45) 
    axes[2*i+1].set_title('Cosine Similarity',size = 9)
    # axes[2*i+1].set_xlabel('Angle')
# fig.tight_layout()
plt.show()

#%% Example location heatmaps, seperately sorted by strength
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
sns.heatmap(sorted_mat.iloc[:,:-1],center = 0,vmax = 2.5,vmin = -1,xticklabels=False,yticklabels=False,ax = ax)
# Corr_Matrix_Norm = Corr_Matrix_Norm.drop(['Best_Angle'],axis = 1)
ax.set_title('Similarity with All Orientation Maps')
ax.set_ylabel('Frames')
ax.set_xticks([0,45,90,135])
ax.set_xticklabels([0,45,90,135])
ax.set_xlabel('Orientation Angles')
fig.tight_layout()
plt.show()
# example_loc_corrs = 
#%% ###################### 2. FOR ALL LOCATIONS ###############################
# All Location Heatmaps, seperately sorted by strength
for i,cloc in enumerate(all_path_dic):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable_v2(cloc,'Spon_Before.pkl')
    c_model = ot.Load_Variable(cloc,'Orien_UMAP_3D_20comp.pkl')
    analyzer = UMAP_Analyzer(ac = ac,umap_model=c_model,spon_frame=c_spon,od = False,color = False)
    analyzer.Train_SVM_Classifier(C = 1)
    spon_labels = analyzer.spon_label
    stim_labels = analyzer.stim_label
    c_corr_frames = all_spon_corr_dic[cloc_name]
    on_parts = c_corr_frames.iloc[spon_labels>0,:]
    off_parts = c_corr_frames.iloc[spon_labels==0,:]
    if i ==0:
        all_on_parts = copy.deepcopy(on_parts)
        all_off_parts = copy.deepcopy(off_parts)
    else:
        all_on_parts = pd.concat([all_on_parts,on_parts])
        all_off_parts = pd.concat([all_off_parts,off_parts])

#%% sort and plot Heatmaps
all_on_parts['Best_Angle'] = all_on_parts.idxmax(1)
on_parts_sorted = all_on_parts.sort_values(by=['Best_Angle'])
on_parts_sorted  = on_parts_sorted.drop(['Best_Angle'],axis = 1)
all_on_parts  = all_on_parts.drop(['Best_Angle'],axis = 1)
all_off_parts['Best_Angle'] = all_off_parts.idxmax(1)
off_parts_sorted = all_off_parts.sort_values(by=['Best_Angle'])
off_parts_sorted  = off_parts_sorted.drop(['Best_Angle'],axis = 1)
all_off_parts  = all_off_parts.drop(['Best_Angle'],axis = 1)
sorted_mat = pd.concat([on_parts_sorted,off_parts_sorted])
plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4),dpi = 180)
sns.heatmap(sorted_mat.iloc[:,:-1],center = 0,vmax = 2.5,vmin = -1,xticklabels=False,yticklabels=False,ax = ax)
# Corr_Matrix_Norm = Corr_Matrix_Norm.drop(['Best_Angle'],axis = 1)
ax.set_title('Similarity with All Orientation Maps (All Locations)')
ax.set_ylabel('Frames')
ax.set_xticks([0,45,90,135])
ax.set_xticklabels([0,45,90,135])
ax.set_xlabel('Orientation Angles')
fig.tight_layout()
plt.show()


#%% All Location Repeat Counts
all_repeats = copy.deepcopy(all_on_parts)
all_repeats['Best_Angle'] = all_repeats.idxmax(1)

plt.clf()
plt.cla()
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
n_bins = 30
rads = np.radians(np.array(all_repeats['Best_Angle'].astype('f8')))*2

ax.set_xticks(np.arange(0, 2*np.pi, 2*np.pi/6))
ax.set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°'])
ax.set_rlim(0,1000)
ax.set_rticks([200,400,600,800])
ax.set_rlabel_position(45) 
# ax.set_xlabel('Repeat Counts')
ax.hist(rads, bins=n_bins,rwidth=1)
ax.set_title('All Orientation Repeat in Spontaneous')
fig.tight_layout()
plt.show()