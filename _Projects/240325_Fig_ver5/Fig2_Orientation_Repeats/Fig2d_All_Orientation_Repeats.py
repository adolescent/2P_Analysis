'''
This will plot the repeat of all orientation maps.

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

all_orien_maps = ot.Load_Variable(work_path,'VAR1_All_Orien_Response.pkl')
all_cell_oriens = ot.Load_Variable(work_path,'VAR2_All_Cell_Best_Oriens.pkl')
all_spon_corr_mat = ot.Load_Variable(work_path,'VAR3_All_Loc_Corr_Matrix.pkl')



#%%########################## FIG 3B CORR HEATMAP - ALL LOCATION ##############################
#%% 1. Seperate On and Off parts
pc_num = 10
for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = np.array(ot.Load_Variable_v2(cloc,'Spon_Before.pkl'))
    _,_,c_model = Z_PCA(Z_frame=c_spon,sample='Frame',pcnum=pc_num)
    analyzer = UMAP_Analyzer(ac = ac,umap_model=c_model,spon_frame=c_spon,od = 0,orien = 1,color = 0,isi = True)
    analyzer.Train_SVM_Classifier(C=1)
    stim_embed = analyzer.stim_embeddings
    stim_label = analyzer.stim_label
    spon_embed = analyzer.spon_embeddings
    spon_label = analyzer.spon_label
    c_corr_frames = all_spon_corr_mat[cloc_name]
    on_parts = c_corr_frames.iloc[spon_label>0,:]
    off_parts = c_corr_frames.iloc[spon_label==0,:]
    if i ==0:
        all_on_parts = copy.deepcopy(on_parts)
        all_off_parts = copy.deepcopy(off_parts)
    else:
        all_on_parts = pd.concat([all_on_parts,on_parts])
        all_off_parts = pd.concat([all_off_parts,off_parts])


#%% 2. sort and plot Heatmaps
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
sns.heatmap(sorted_mat.iloc[:,:-1],center = 0,vmax = 1,vmin = -1,xticklabels=False,yticklabels=False,ax = ax)
# Corr_Matrix_Norm = Corr_Matrix_Norm.drop(['Best_Angle'],axis = 1)
ax.set_title('Similarity with All Orientation Maps (All Locations)')
ax.set_ylabel('Frames')
ax.set_xticks([0,45,90,135])
ax.set_xticklabels([0,45,90,135])
ax.set_xlabel('Orientation Angles')
sns.lineplot(x=[0,45,90,135,180], y=len(all_on_parts),color = 'y')
fig.tight_layout()
plt.show()




#%% ################### FIG S3B - ALL LOC G16 CORR MATRIX ########################
# shows the cosine similarity on G16 stims' 4 orien and isi.
#%% 1. Get G16 Corr Matrix.

all_g16_labels = []
for i,cloc in enumerate(all_path_dic):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    g16_frame,g16_label = ac.Combine_Frame_Labels(od = 0,orien = 1,color = 0,isi = 1)
    all_g16_labels.extend(g16_label)
    c_stim_maps = all_orien_maps[cloc_name]
    c_corr_mat = pd.DataFrame(0.0,columns = c_stim_maps.columns,index = range(len(g16_frame)))
    for j in tqdm(range(len(g16_frame))):
        single_spon = np.array(g16_frame)[j,:]
        for k in range(c_stim_maps.shape[1]):
            c_pattern = np.array(c_stim_maps.iloc[:,k])
            cos_sim = single_spon.dot(c_pattern) / (np.linalg.norm(single_spon) * np.linalg.norm(c_pattern))
            c_corr_mat.iloc[j,k] = cos_sim
    if i == 0:
        all_g16_maps = copy.deepcopy(c_corr_mat)
    else:
        all_g16_maps = pd.concat([all_g16_maps,c_corr_mat])

#%% 2. select Stim ON parts and stim off parts.

on_parts,on_labels = Select_Frame(all_g16_maps,all_g16_labels,[9,11,13,15])
off_parts,_ = Select_Frame(all_g16_maps,all_g16_labels,[0])
sorter = pd.DataFrame(on_parts)
sorter['Index'] = on_labels
on_parts_sorted = sorter.sort_values(by=['Index'])
on_parts_sorted  = on_parts_sorted.drop(['Index'],axis = 1)
off_parts = pd.DataFrame(off_parts)

#%% 3. Plot G16 Heatmaps
plt.clf()
plt.cla()
plotable_mat = pd.concat([on_parts_sorted,off_parts])
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4),dpi = 180)
sns.heatmap(plotable_mat.iloc[:,:-1],center = 0,vmax = 1,vmin = -1,xticklabels=False,yticklabels=False,ax = ax)
# Corr_Matrix_Norm = Corr_Matrix_Norm.drop(['Best_Angle'],axis = 1)
ax.set_title('G16 Similarity with All Orientation Maps (All Locations)')
ax.set_ylabel('Frames')
ax.set_xticks([0,45,90,135])
ax.set_xticklabels([0,45,90,135])
ax.set_xlabel('Orientation Angles')
sns.lineplot(x=[0,45,90,135,180], y=len(on_parts_sorted),color = 'y')
fig.tight_layout()
plt.show()



#%% #################### FIG 3C - COUNT ALL REPEAT NUMS ###############################
# all_repeats = copy.deepcopy(all_on_parts)
# all_repeat_parts = pd.concat([all_on_parts,all_off_parts])
# all_repeat_parts = all_repeat_parts[all_repeat_parts.max(1)>0.5] # define best corr
# all_repeats = copy.deepcopy(all_on_parts)

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
ax.set_title('All Orientation Repeat in Spontaneous',size = 14)
fig.tight_layout()
plt.show()