'''
Whether we have new ensembles?
Try to determine which ensemble of these ones be like.
Try Different methods.

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
from Filters import Signal_Filter

work_path = r'D:\_Path_For_Figs\FigS2e_All_Orientation_Funcmap'
expt_folder = r'D:\_All_Spon_Datas_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
ac.wp = expt_folder
# Get stim label and stim response.
all_stim_frame,all_stim_label = ac.Combine_Frame_Labels(od = True,orien = True,color = True,isi = True)
spon_series = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
reducer = ot.Load_Variable(expt_folder,'All_Stim_UMAP_3D_20comp.pkl')
#%%################# METHOD 1, ALL UMAP ON ENSEMBLES ##########################
#%% 1. Get all spon repeat frames
stim_embeddings = reducer.transform(all_stim_frame)
spon_embeddings = reducer.transform(spon_series)
classifier,score = SVM_Classifier(embeddings=stim_embeddings,label = all_stim_label)
predicted_spon_label = SVC_Fit(classifier,data = spon_embeddings,thres_prob = 0)
repeat_frames = spon_series.iloc[np.where(predicted_spon_label>0)[0],:]
all_repeat_label = predicted_spon_label[np.where(predicted_spon_label>0)[0]]
#%% 2. Calculate Correlation with spon to all stim pattern, and calculate a similarity matrix.
# get all stim pattern first
stim_response_matrix = pd.DataFrame(0.0,columns=range(1,23),index = ac.acn)
for c_stim in range(1,23): # get all stim maps.
    c_id = np.where(all_stim_label == c_stim)[0]
    c_stim_response = all_stim_frame.iloc[c_id,:].mean(0)
    stim_response_matrix.loc[:,c_stim] = c_stim_response
# then corr with stim and spon.
corr_matrix = np.zeros(shape = (22,len(repeat_frames)),dtype = 'f8')#0-7OD,8-15Ori,16-21Col
for i,c_frame in enumerate(np.array(repeat_frames)):
    for j in range(22):
        cc_stimmap = stim_response_matrix.loc[:,j+1]
        c_r,_ = pearsonr(c_frame,cc_stimmap)
        corr_matrix[j,i] = c_r
# adjust corr matrix, generate Frame_Network function.
best_similarity = pd.DataFrame(0.0,columns=['OD','Orien','Color','Low_Corr'],index = range(len(all_repeat_label)))
best_similarity['OD'] = corr_matrix[:8].max(0)
best_similarity['Orien'] = corr_matrix[8:16].max(0)
best_similarity['Color'] = corr_matrix[16:].max(0)
best_similarity['Low_Corr'] = corr_matrix[:].max(0)
#%% 3. Sort graph.
similar_thres = 0.2
best_similarity_sorted = copy.deepcopy(best_similarity)
best_similarity_sorted['Sorting_Index'] = 0.0
best_similar_map = best_similarity.iloc[:,:3].idxmax(1)
best_similar_corr = best_similarity.iloc[:,:3].max(1)
for i in range(len(best_similar_corr)):
    if best_similar_corr[i]<similar_thres:
        best_similarity_sorted.loc[i,'Sorting_Index'] = best_similar_corr[i]
    else:
        best_similarity_sorted.loc[i,'Low_Corr'] = 0
        if best_similar_map[i] == 'OD':
            best_similarity_sorted.loc[i,'Sorting_Index'] = best_similar_corr[i]+3
        elif best_similar_map[i] == 'Orien':
            best_similarity_sorted.loc[i,'Sorting_Index'] = best_similar_corr[i]+2
        elif best_similar_map[i] == 'Color':
            best_similarity_sorted.loc[i,'Sorting_Index'] = best_similar_corr[i]+1
best_similarity_sorted = best_similarity_sorted.sort_values(['Sorting_Index'], ascending=[False])
best_similarity_sorted = best_similarity_sorted.drop('Sorting_Index',axis = 1)
plt.clf()
plt.cla()
fig, ax = plt.subplots(figsize=(2.5,9),dpi = 180)
cbar_ax = fig.add_axes([.96, .4, .05, .2])
sns.heatmap(best_similarity_sorted,center = 0, ax=ax,yticklabels=False,cbar_ax = cbar_ax)
ax.set_title('Spon Repeats Similarity vs Stim Map',size = 8)
fig.tight_layout()
#%% 4. Compre Repeat and Non Repeat Activation strength.
non_repeat_locs = np.where(best_similarity.max(1)<similar_thres)[0]# not repeat any map
non_repeat_frames = np.array(repeat_frames)[non_repeat_locs,:] # non repeat cell response 

stim_repeat_locs = np.where(best_similarity.max(1)>similar_thres)[0] # repeat atleast a map.
stim_repeat_frames = np.array(repeat_frames)[stim_repeat_locs,:] # repeat cell response 
# count type 1, count avr response directly.
non_repeat_response = non_repeat_frames.mean(1)
repeat_response = stim_repeat_frames.mean(1)
sample_size = len(non_repeat_response)
selected_stim_response = np.random.choice(repeat_response,sample_size)

fig, ax = plt.subplots(figsize=(6,4),dpi = 180)
ax.hist(selected_stim_response,bins = 20,alpha = 0.7,label = 'Known Ensemble')
ax.hist(non_repeat_response,bins = 20,alpha = 0.7,label = 'Unknown Ensemble')
ax.legend()
ax.set_title('Compare of Stim Ensemble and Unknown Ensemble')
#%% count type 2, count 'ON' neuron numbers.(Need a threshold)
on_thres = 1
thresed_repeat_frames = stim_repeat_frames>on_thres
thresed_nonrepeat_frames = non_repeat_frames>on_thres
thresed_scale_repeat = thresed_repeat_frames.sum(1)
thresed_scale_nonrepeat = thresed_nonrepeat_frames.sum(1)

sample_size = len(thresed_scale_nonrepeat)
selected_stim_response = np.random.choice(thresed_scale_repeat,sample_size)
fig, ax = plt.subplots(figsize=(6,4),dpi = 180)
ax.hist(selected_stim_response,bins = 20,alpha = 0.7,label = 'Known Ensemble')
ax.hist(thresed_scale_nonrepeat,bins = 20,alpha = 0.7,label = 'Unknown Ensemble')
ax.legend()
ax.set_title('Compare of Stim and Unknown Ensemble Size')
#%% 5. Show several example of non-repeat maps.
value_max = 6
value_min = -3
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(4,8),dpi = 180)
cbar_ax = fig.add_axes([.97, .25, .05, .5])
for i in range(2):
    for j in range(4):
        c_example = non_repeat_frames[np.random.randint(0,len(non_repeat_frames)),:]
        c_example_graph = ac.Generate_Weighted_Cell(c_example)
        sns.heatmap(c_example_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[j,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
fig.suptitle('Example of Non Repeat Frames')
fig.tight_layout()
plt.show()
# And all non-repeat Average
global_avr = non_repeat_frames.mean(0)
global_avr_graph = ac.Generate_Weighted_Cell(global_avr)
fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(4,4),dpi = 180)
sns.heatmap(global_avr_graph,center = 0,xticklabels=False,yticklabels=False,ax = ax2,vmax = 2,vmin = -1,square=True)
fig2.suptitle('Average of Non Repeats')
fig2.tight_layout()
plt.show()

#%%################# METHOD 2, STIM TRAIN ANALYSIS ##########################
thres = -99 # below thres will be set as 0.
