'''
This script will align time according to y coordinate, and remake the umap to see whether expression will change.


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
import seaborn as sns
from My_Wheels.Cell_Class.Stim_Calculators import Stim_Cells
from My_Wheels.Cell_Class.Format_Cell import Cell
from Cell_Class.Advanced_Tools import *
from Cell_Class.Plot_Tools import *

wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220420_L91\_CAIMAN'
# wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220914_L85_2P\_CAIMAN'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
# reducer = ot.Load_Variable(wp,r'Stim_No_ISI_UMAP_Unsup_3d.pkl')
spon_frame_raw = ac.Z_Frames['1-001']
od_frame_raw = ac.Z_Frames[ac.odrun]
orien_frame_raw = ac.Z_Frames[ac.orienrun]
#%% Define shift function.
def Shifter(series,y_corr): 
    move_weight = y_corr/512
    shifted_series = np.zeros(len(series))
    for i in range(1,len(series)):
        cf = series[i-1]*move_weight+series[i]*(1-move_weight)
        shifted_series[i] = cf
    return shifted_series # id 0 is 0 for no previous data.
#%% Get moved series of each frame.
spon_frame_shifted = pd.DataFrame(0,columns = spon_frame_raw.columns,index=spon_frame_raw.index)
od_frame_shifted = pd.DataFrame(0,columns = od_frame_raw.columns,index=od_frame_raw.index)
orien_frame_shifted = pd.DataFrame(0,columns = orien_frame_raw.columns,index=orien_frame_raw.index)
for i,cc in tqdm(enumerate(ac.acn)):
    c_spon_series = np.array(spon_frame_raw[cc])
    c_orien_series = np.array(orien_frame_raw[cc])
    c_od_series = np.array(od_frame_raw[cc])
    cc_y = ac.Cell_Locs[cc]['Y']
    # write each shifted frame.
    spon_frame_shifted[cc] = Shifter(c_spon_series,cc_y)
    orien_frame_shifted[cc] = Shifter(c_orien_series,cc_y)
    od_frame_shifted[cc] = Shifter(c_od_series,cc_y)
#%% train space using shifted frame.
_,all_label = ac.Combine_Frame_Labels(isi = True)# sequence:OD+Orien
all_frame = pd.concat([od_frame_shifted,orien_frame_shifted]).reset_index(drop= True)
# cut all isi
non_isi = np.where(all_label != 0)[0]
non_isi_label = all_label[non_isi]
non_isi_frame = all_frame.loc[non_isi,:]
unshift_frame,unshift_label = ac.Combine_Frame_Labels()
#%% UMAP on shifted data.
reducer = umap.UMAP(n_components=3,n_neighbors=20)
reducer.fit(non_isi_frame)
u = reducer.embedding_
plt.switch_backend('webAgg')
fig,ax = Plot_3D_With_Labels(data = u,labels = non_isi_label)
plt.show()
# Save_3D_Gif(ax,fig)
#%% Do svm cluster on 2 kind of data and embedding spon.
spon_embedding_origin = reducer.transform(spon_frame_raw)
spon_embedding_shifted = reducer.transform(spon_frame_shifted)
#%% Plot 2 kind of spon results.
plt.switch_backend('webAgg')
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
ax.grid(False)
ax.scatter3D(spon_embedding_origin[:,0],spon_embedding_origin[:,1],spon_embedding_origin[:,2],s = 5,color = 'r')
ax.scatter3D(spon_embedding_shifted[:,0],spon_embedding_shifted[:,1],spon_embedding_shifted[:,2],s = 5,color = 'b')
plt.show()
#%% Do svm cluster on shifted data and real data.
classifier = SVM_Classifier(u,non_isi_label,C = 10)
predicted_label = SVC_Fit(classifier,spon_embedding_shifted,thres_prob=0.45)
plt.switch_backend('webAgg')
fig,ax = Plot_3D_With_Labels(data = spon_embedding_shifted,labels = predicted_label)
plt.show()
#%% average all label and generate graph.
all_response = Average_Each_Label(spon_frame_raw,predicted_label)
plt.switch_backend('webAgg')
fig,ax = Plot_Multi_Subgraphs(all_response,ac.all_cell_dic,(4,5))
plt.show()
#%% Seperate cell.
reducer = umap.UMAP(n_neighbors = 30,n_components = 2,min_dist= 0.1,random_state=621)
reducer.fit(spon_frame_shifted.T)
u = reducer.embedding_
# plots
OD_index = ac.all_cell_tunings.loc['OD',:]
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
ax = plt.scatter(x = u[:,0],y = u[:,1],c = OD_index,cmap = 'bwr')
plt.show()

#%% network alteration after time align.
import networkx as nx
# make sure we use right labels.
data = predicted_label
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

