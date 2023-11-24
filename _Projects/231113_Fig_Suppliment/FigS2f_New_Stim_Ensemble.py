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
'''
This part will try to generate strength plot of all known network and 3 label network.
'''
#%% Step1, get all tuning index response.

thres = -99 # below thres will be set as 0.
thresed_all_spon_response = copy.deepcopy(spon_series)
thresed_all_spon_response[thresed_all_spon_response<thres] = 0
def Submap_mask_Generator(input_map):
    c_A_response = input_map.loc['A_reponse',:]
    c_B_response = input_map.loc['B_response',:]
    c_submap = (c_A_response-c_B_response)# Raw-0 submaps.
    weighted_mask = c_submap/c_submap.sum()
    return weighted_mask

Score_Frame = pd.DataFrame(0.0,index = range(len(thresed_all_spon_response)),columns=['L','R','Orien0','Orien45','Orien90','Orien135','Red','Green','Blue'])
LE_mask = Submap_mask_Generator(ac.OD_t_graphs['L-0'])
Score_Frame['L'] = np.dot(thresed_all_spon_response,LE_mask)
RE_mask = Submap_mask_Generator(ac.OD_t_graphs['R-0'])
Score_Frame['R']  = np.dot(thresed_all_spon_response,RE_mask)
Orien0_mask = Submap_mask_Generator(ac.Orien_t_graphs['Orien0-0'])
Score_Frame['Orien0']  = np.dot(thresed_all_spon_response,Orien0_mask)
Orien45_mask = Submap_mask_Generator(ac.Orien_t_graphs['Orien45-0'])
Score_Frame['Orien45']  = np.dot(thresed_all_spon_response,Orien45_mask)
Orien90_mask = Submap_mask_Generator(ac.Orien_t_graphs['Orien90-0'])
Score_Frame['Orien90']  = np.dot(thresed_all_spon_response,Orien90_mask)
Orien135_mask = Submap_mask_Generator(ac.Orien_t_graphs['Orien135-0'])
Score_Frame['Orien135']  = np.dot(thresed_all_spon_response,Orien135_mask)
Red_mask = Submap_mask_Generator(ac.Color_t_graphs['Red-0'])
Score_Frame['Red']  = np.dot(thresed_all_spon_response,Red_mask)
Green_mask = Submap_mask_Generator(ac.Color_t_graphs['Green-0'])
Score_Frame['Green']  = np.dot(thresed_all_spon_response,Green_mask)
Blue_mask = Submap_mask_Generator(ac.Color_t_graphs['Blue-0'])
Score_Frame['Blue']  = np.dot(thresed_all_spon_response,Blue_mask)
ot.Save_Variable(work_path,'Stim_Map_Score',Score_Frame)
#%% step2, try to sort graph, see the difference.
score_thres = 0.5 # below this score will be set as Null.
Score_Frame_Sorted = copy.deepcopy(Score_Frame)
# Score_Frame_Sorted['Strength'] = np.array(thresed_all_spon_response.sum(1))
# Score_Frame_Sorted = Score_Frame_Sorted.sort_values(['Strength'], ascending=[False])
# Score_Frame_Sorted = Score_Frame_Sorted.drop('Strength',axis = 1)
Score_Frame_Sorted['Eye Score'] = Score_Frame.iloc[:,:2].max(1)
Score_Frame_Sorted['Orien Score'] = Score_Frame.iloc[:,2:6].max(1)
Score_Frame_Sorted['Color Score'] = Score_Frame.iloc[:,6:].max(1)
Score_Frame_Sorted['Response'] = np.array(thresed_all_spon_response.mean(1))
Score_Frame_Sorted = Score_Frame_Sorted[['Eye Score','Orien Score','Color Score']]
Score_Frame_Sorted['Sorting Index'] = 0.0
all_response = np.array(thresed_all_spon_response.mean(1))
for i in range(len(Score_Frame_Sorted)):
    c_umap_label = predicted_spon_label[i]
    # max_tuning = Score_Frame_Sorted.loc[i,:].max()
    # if max_tuning < score_thres:
    #     Score_Frame_Sorted.loc[i,'Sorting Index'] = all_response[i]
    # else:
    #     max_type = Score_Frame_Sorted.loc[i,:].idxmax()
    #     if max_type == 'Eye Score':
    #         Score_Frame_Sorted.loc[i,'Sorting Index'] = all_response[i]+30 # OD+3 
    #     elif max_type == 'Orien Score':
    #         Score_Frame_Sorted.loc[i,'Sorting Index'] = all_response[i]+20 # Orien+2 
    #     elif max_type == 'Color Score':
    #         Score_Frame_Sorted.loc[i,'Sorting Index'] = all_response[i]+10 # Color+2 
    if c_umap_label>0 and c_umap_label<9:# OD repeats
        Score_Frame_Sorted.loc[i,'Sorting Index'] = all_response[i]+30
    elif c_umap_label>8 and c_umap_label<17:
        Score_Frame_Sorted.loc[i,'Sorting Index'] = all_response[i]+20
    elif c_umap_label>16:
        Score_Frame_Sorted.loc[i,'Sorting Index'] = all_response[i]+10
    else:
        Score_Frame_Sorted.loc[i,'Sorting Index'] = all_response[i]


Score_Frame_Sorted['Response'] = np.array(thresed_all_spon_response.mean(1))
Score_Frame_Sorted = Score_Frame_Sorted.sort_values(['Sorting Index'], ascending=[False])
# Score_Frame_Sorted = Score_Frame_Sorted.sort_values(['Response'], ascending=[False])
Score_Frame_Sorted = Score_Frame_Sorted.drop('Sorting Index',axis = 1)
#%% 3. Plot all orientation score.
thres_similar = 0.2 # this thres is similarity with stim patterns.
corr_matrix_all = np.zeros(shape = (22,len(spon_series)),dtype = 'f8')#0-7OD,8-15Ori,16-21Col
for i,c_frame in tqdm(enumerate(np.array(spon_series))):
    for j in range(22):
        cc_stimmap = stim_response_matrix.loc[:,j+1]
        c_r,_ = pearsonr(c_frame,cc_stimmap)
        corr_matrix_all[j,i] = c_r
# plot tuning score 
plt.clf()
plt.cla()
fig, ax = plt.subplots(figsize=(4,12),dpi = 180)
sns.heatmap(Score_Frame_Sorted,center = 0,vmax = 3,ax = ax,yticklabels=False)
# and add umap class information.
sorted_index = Score_Frame_Sorted.index
all_best_corr = corr_matrix_all.max(0)
for i,c_label in enumerate(predicted_spon_label):
    phy_loc = np.where(sorted_index == i)[0][0]
    c_max_corr = all_best_corr[i]
    if c_max_corr<thres_similar and c_label>0:
        ax.scatter(x = [3],y = [phy_loc],s=0.5,color = 'black')
    else:
        if c_label>0 and c_label<9:
            ax.scatter(x = [3],y = [phy_loc],s=0.5,color = 'y')
        elif c_label>8 and c_label<17:
            ax.scatter(x = [3],y = [phy_loc],s=0.5,color = 'g')
        elif c_label>16:
            ax.scatter(x = [3],y = [phy_loc],s=0.5,color = 'b')
# ax.scatter(x = [3,3],y = [300,2505],s=3,color = 'y',label = 'Eye Repeat')
ax.set_title('Tuning Scores')
# sns.move_legend(ax,  "upper left", bbox_to_anchor=(1,0.8))
# calculate umap propotion of all response frames.
positive_ids = np.array(Score_Frame_Sorted[Score_Frame_Sorted['Response']>1].index)
umap_nums = (predicted_spon_label[positive_ids]>0).sum()
print(f'UMAP Seperated Stim Cell Number {umap_nums} / {len(positive_ids)}, with propotion {umap_nums*100/len(positive_ids):.2f}')
#%%4. Example of Response good, but umap ignored frame.
example_frame = Score_Frame_Sorted.index[1243]
example_frame2 = Score_Frame_Sorted.index[1244]
cc_spon = spon_series.iloc[example_frame,:]
cc_spon2 = spon_series.iloc[example_frame2,:]
cc_spon_graph = ac.Generate_Weighted_Cell(cc_spon)
cc_spon_graph2 = ac.Generate_Weighted_Cell(cc_spon2)

value_max = 6
value_min = -3
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4,8),dpi = 180)
cbar_ax = fig.add_axes([.97, .25, .05, .5])
sns.heatmap(cc_spon_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
sns.heatmap(cc_spon_graph2,center = 0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
fig.suptitle('Example of UMAP Unclassified Frame')
fig.tight_layout()
plt.show()


