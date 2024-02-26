'''
This script will answer the question of how network alteration inside spon?
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

wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220630_L76_2P\_CAIMAN'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
# reducer = ot.Load_Variable(wp,'Stim_No_ISI_UMAP_Unsup_3d.pkl')
# u = reducer.embedding_
#%% Get G16/OD data and labels, use them to train an UMAP.
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
#%% Combine two data frame with label.
all_stim_frame = pd.concat([od_noisi,orien_noisi]).reset_index(drop = True)
# all_stim_frame = pd.concat([all_stim_frame,color_noisi]).reset_index(drop = True)
#generate all real labels. Use 1357 as LE,2468 as RE and 9-16 as Orien,0 as all ISI.
label_od_finale = np.array(od_label_noisi['Raw_ID'])
label_orien_finale = np.zeros(len(orien_label_noisi),dtype = 'i4')
for i in range(len(label_orien_finale)):
    c_label = orien_label_noisi.iloc[i,1]
    if c_label == 0:
        label_orien_finale[i] = 0
    elif c_label == 8 or c_label == 16:
        label_orien_finale[i] = 8+8
    else:
        label_orien_finale[i] = 8+c_label%8
# label_color_finale =  np.array(color_label_noisi['Raw_ID']%7+17)
# all_label = np.hstack([label_od_finale,label_orien_finale,label_color_finale])
all_label = np.hstack([label_od_finale,label_orien_finale])
#%% we can do umap here.
reducer = umap.UMAP(n_neighbors = 20,min_dist=0.01,n_components=3)
reducer.fit(all_stim_frame)
u = reducer.embedding_
# ot.Save_Variable(ac.wp,'Stim_No_ISI_UMAP_Unsup_3d',reducer) # done.
# Plot 3D
plt.switch_backend('webAgg')
fig,ax = Plot_3D_With_Labels(data = u,labels = all_label)
plt.show()
#%% umap spon data on this space.
spon_embeddings = reducer.transform(spon_frames)
#%% Plot 3D stack with spon.
plt.switch_backend('webAgg')
fig,ax = Plot_3D_With_Labels(data = u,labels = all_label)
ax.scatter3D(spon_embeddings[:,0],spon_embeddings[:,1],spon_embeddings[:,2],c = 'r',s = 5)
plt.show()
#%% Establish SVM Classifier.
classifier = SVM_Classifier(embeddings = u,label = all_label,C = 10)
spon_label_svc = SVC_Fit(classifier,spon_embeddings,thres_prob = 0.5)
spon_label_svc[spon_label_svc == 0] = -1
plt.switch_backend('webAgg')
fig,ax = Plot_3D_With_Labels(data = spon_embeddings,labels = spon_label_svc)
plt.show()
#%% Try to plot a directed map.
import networkx as nx
# make sure we use right labels.
data = spon_label_svc
# define a vanilla nx graph.
G = nx.Graph()
# add all labels as nodes.
for value in set(data):
    G.add_node(value)
pos = nx.spring_layout(G) 
# set position of each point manually.
pos[1]=(0,2)
pos[2]=(20,2)
pos[3]=(-2,0)
pos[4]=(18,0)
pos[5]=(0,-2)
pos[6]=(22,0)
pos[7]=(2,0)
pos[8]=(20,-2)
pos[9]=(10,2)
pos[10]=(10-1.4,1.4)
pos[11]=(8,0)
pos[12]=(10-1.4,-1.4)
pos[13]=(10,-2)
pos[14]=(11.4,-1.4)
pos[15]=(12,0)
pos[16]=(11.4,1.4)
pos[-1]=(10,-8)
# Add edges to the graph and assign weights based on the frequency of each edge
edge_weights = {}
for i in range(len(data)-1):
    edge = (data[i], data[i+1])
    edge_rev = (data[i+1], data[i])
    if (edge in edge_weights):# make both side conenction even.
        edge_weights[edge] += 1
    elif (edge_rev in edge_weights):
        edge_weights[edge_rev] += 1
    else:
        edge_weights[edge] = 1
    G.add_edge(*edge)
# clip edge_weights to avoid null of all results.
max_weight = 100
for i,c_link in enumerate(list(edge_weights.keys())):
    edge_weights[c_link] = min(edge_weights[c_link],max_weight)
# Set the weights of the edges
for edge, weight in edge_weights.items():
    G[edge[0]][edge[1]]['weight'] = weight

# Draw the graph with edge weights as edge colors
edge_colors = [edge_data['weight'] for _, _, edge_data in G.edges(data=True)]

plt.switch_backend('webAgg')
nx.draw_networkx(G, pos, with_labels=True, edge_color=edge_colors, edge_cmap=plt.cm.Blues)
plt.show()