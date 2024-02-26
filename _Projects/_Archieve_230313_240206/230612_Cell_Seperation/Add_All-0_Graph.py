'''
What will the result be with all-0?
I'll make several adjusted all-0 graph into UMAP generation.
Let's see if there are all-0 graphs in it.
'''



#%%
import OS_Tools_Kit as ot
import numpy as np
import pandas as pd
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
import Graph_Operation_Kit as gt
import cv2
import umap
import umap.plot
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from Kill_Cache import kill_all_cache
from sklearn.model_selection import cross_val_score
from sklearn import svm
from Cell_Tools.Cell_Visualization import Cell_Weight_Visualization
import random
from My_Wheels.Cell_Class.Stim_Calculators import Stim_Cells
from My_Wheels.Cell_Class.Format_Cell import Cell
from Cell_Class.Advanced_Tools import *
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Kill_Cache import kill_all_cache
from Cell_Class.Plot_Tools import *
wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220420_L91\_CAIMAN'
caiman_path = r'C:\ProgramData\anaconda3\envs\umapzr'
kill_all_cache(caiman_path)
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
od_frames = ac.Z_Frames[ac.odrun]
orien_frames = ac.Z_Frames[ac.orienrun]
color_frames = ac.Z_Frames[ac.colorrun]
od_labels = ac.OD_Frame_Labels
orien_labels = ac.Orien_Frame_Labels
color_labels = ac.Color_Frame_Labels
od_noisi,od_label_noisi = Remove_ISI(od_frames,od_labels)
orien_noisi,orien_label_noisi = Remove_ISI(orien_frames,orien_labels)
color_noisi,color_label_noisi = Remove_ISI(color_frames,color_labels)
spon_frames = ac.Z_Frames['1-001']
#%% Get L-0,R-0,All-0 Maps
od_t_graphs = ac.OD_t_graphs
orien_t_graph = ac.Orien_t_graphs
L_All_map = od_t_graphs['L-0'].loc['A_reponse']
R_All_map = od_t_graphs['R-0'].loc['A_reponse']
All_map = orien_t_graph['All-0'].loc['A_reponse']
# L_All_graph = ac.Generate_Weighted_Cell(L_All_map)
# R_All_graph = ac.Generate_Weighted_Cell(R_All_map)
# All_graph = ac.Generate_Weighted_Cell(All_map)
#%% Generate variations.
noise_level = 0.5
variation_num = 30
L_Maps = pd.DataFrame(0,index = range(30),columns=ac.acn)
R_Maps = pd.DataFrame(0,index = range(30),columns=ac.acn)
All_Maps = pd.DataFrame(0,index = range(30),columns=ac.acn)

for i in range(variation_num):
    rand_noise = np.random.rand(ac.cellnum)*noise_level-noise_level/2
    c_L = L_All_map+rand_noise
    L_Maps.loc[i,:] = c_L
    rand_noise = np.random.rand(ac.cellnum)*noise_level-noise_level/2
    c_R = R_All_map+rand_noise
    R_Maps.loc[i,:] = c_R
    rand_noise = np.random.rand(ac.cellnum)*noise_level-noise_level/2
    c_A = All_map+rand_noise
    All_Maps.loc[i,:] = c_A
#%% Get all labeled_stim map and add manually generated maps.
# all_stim_frame = pd.concat([od_noisi,orien_noisi]).reset_index(drop = True)
all_stim_frame = pd.concat([od_frames,orien_frames]).reset_index(drop=True)
# all_stim_frame = pd.concat([all_stim_frame,color_noisi]).reset_index(drop = True)
#generate all real labels. Use 1357 as LE,2468 as RE and 9-16 as Orien,0 as all ISI.
label_od_finale = np.array(od_labels.T['Raw_ID'])
label_od_finale[label_od_finale == -1]=0
label_orien_finale = np.zeros(len(orien_labels.T),dtype = 'i4')
for i in range(len(label_orien_finale)):
    c_label = orien_labels.T.iloc[i,1]
    if c_label == 0 or c_label ==-1:
        label_orien_finale[i] = 0
    elif c_label == 8 or c_label == 16:
        label_orien_finale[i] = 8+8
    else:
        label_orien_finale[i] = 8+c_label%8
# label_color_finale =  np.array(color_label_noisi['Raw_ID']%7+17)
# all_label = np.hstack([label_od_finale,label_orien_finale,label_color_finale])
all_label = np.hstack([label_od_finale,label_orien_finale])
# Add labels with human-made stims.
all_stim_frame = pd.concat([all_stim_frame,L_Maps]).reset_index(drop = True)
all_stim_frame = pd.concat([all_stim_frame,R_Maps]).reset_index(drop = True)
all_stim_frame = pd.concat([all_stim_frame,All_Maps]).reset_index(drop = True)
all_label = np.append(all_label, np.full(30, 17))
all_label = np.append(all_label, np.full(30, 18))
all_label = np.append(all_label, np.full(30, 19))
#%% UMAP
reducer = umap.UMAP(n_neighbors = 20,min_dist=0.01,n_components=3,target_weight=0)
reducer.fit(all_stim_frame,all_label)
u = reducer.embedding_
# ot.Save_Variable(ac.wp,'Stim_All_Add_UMAP_Unsup_3d',reducer) # done.
# Plot 3D
plt.switch_backend('webAgg')
fig,ax = Plot_3D_With_Labels(data = u,labels = all_label)
plt.show()
#%% Embedding spon in it.
spon_embeddings = reducer.transform(spon_frames)

plt.switch_backend('webAgg')
fig,ax = Plot_3D_With_Labels(data = u,labels = all_label)
ax.scatter3D(spon_embeddings[:,0],spon_embeddings[:,1],spon_embeddings[:,2],c = 'r',s = 5)
plt.show()
#%% Establish SVM Classifier.
classifier = SVM_Classifier(embeddings = u,label = all_label,C = 5)
spon_label_svc = SVC_Fit(classifier,spon_embeddings,thres_prob = 0.4)
spon_label_svc[spon_label_svc == 0] = 0
plt.switch_backend('webAgg')
fig,ax = Plot_3D_With_Labels(data = spon_embeddings,labels = spon_label_svc)
plt.show()

#%% adjust label, only distinguish L/R/orien.
adj_label = copy.deepcopy(all_label)
adj_label[(adj_label>0)*(adj_label<9)*(adj_label%2==1)]=17 # LE
adj_label[(adj_label>0)*(adj_label<9)*(adj_label%2==0)]=18 # RE
#%% Establish SVM Classifier.
classifier = SVM_Classifier(embeddings = u,label = adj_label,C = 5)
spon_label_svc = SVC_Fit(classifier,spon_embeddings,thres_prob = 0.5)
spon_label_svc[spon_label_svc == 0] = 0
plt.switch_backend('webAgg')
fig,ax = Plot_3D_With_Labels(data = spon_embeddings,labels = spon_label_svc)
plt.show()
'''
This is not a good idea to add all-0, only stim will work better.

'''
#%% recover maps.
all_avr_spon_clust = Average_Each_Label(spon_frames,spon_label_svc)
plt.switch_backend('webAgg')
fig,ax = Plot_Multi_Subgraphs(graph_frame = all_avr_spon_clust,acd = ac.all_cell_dic,shape = (5,4))
plt.show()