'''
This script will procude labels for V4 cell response map.
Let's see whether there are possible clusters.
'''
#%% import 
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

graph_path = r'D:\ZR\_Data_Temp\_Temp_Data\220711_temp\V4 feature clust\Featuremap_set_V4' # should be 385 png file.
csv_path = r'D:\ZR\_Data_Temp\_Temp_Data\220711_temp\V4 feature clust\Cell_Locs.csv'
condapath = r'C:\ProgramData\anaconda3\envs\umapzr'
wp = r'D:\ZR\_Data_Temp\_Temp_Data\220711_temp\V4 feature clust'
kill_all_cache(condapath)
#%% If load in, run this.
reducer = ot.Load_Variable(wp,'V4_unsup_best_stim.pkl')
reducer3d = ot.Load_Variable(wp,'V4_unsup_best_stim_3d')
#%% read in data in properate mode.
# get all file and cell label.
all_graph_name = ot.Get_File_Name(graph_path,'.png')
frame_num = len(all_graph_name)
all_graph_frame = np.zeros(shape = (frame_num,270000),dtype = 'f8')
for i,c_name in enumerate(all_graph_name):
    c_graph = cv2.imread(c_name)# remember cv2 use bgr sequense. But no effect here.
    all_graph_frame[i,:] = c_graph.flatten().astype('f8')

#%% get all cell label with id of name.
data = pd.read_csv(csv_path)# fill site in 2 site and cell in 3 site, get a name.
labels = np.array(data['Domain'])
#%% Try to umap them.
reducer = umap.UMAP(n_neighbors = 20,min_dist=0.01,n_components=2)
reducer.fit(all_graph_frame) # supervised learning on G16 data.
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
umap.plot.points(reducer,ax = ax,labels = labels,theme = 'fire')
plt.show()
ot.Save_Variable(wp,'V4_unsup_best_stim',reducer)
#%% embedding quality
plt.switch_backend('webAgg')
# fig,ax = plt.subplots(1,figsize = (12,12))
fig,ax = plt.subplots(1,figsize = (7,5))
umap.plot.diagnostic(reducer, diagnostic_type='local_dim',ax = ax)
plt.show()
#%% connectivity information.
plt.switch_backend('webAgg')
# fig,ax = plt.subplots(1,figsize = (12,12))
# fig,ax = plt.subplots(1,figsize = (7,5))
umap.plot.connectivity(reducer, edge_bundling='hammer')
plt.show()
#%% embedding data in 3d.
reducer3d = umap.UMAP(n_neighbors = 20,min_dist=0.01,n_components=3)
reducer3d.fit(all_graph_frame) # supervised learning on G16 data.
# reducer.fit(g16_datasets) # unsupervised learning on G16 data.
u = reducer3d.embedding_
ot.Save_Variable(wp,'V4_unsup_best_stim_3d',reducer3d)
#%% 3D plot
plt.clf()
plt.switch_backend('webAgg')
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
ax.grid(False)
# ax.axis('off')
unique_labels = np.unique(labels)
handles = []
all_scatters = []
for label in unique_labels:
    mask = labels == label
    scatter = ax.scatter3D(u[:,0][mask], u[:,1][mask], u[:,2][mask], label=label,s = 15)
    all_scatters.append(scatter)
    handles.append(scatter)
ax.legend(handles=handles)
plt.show()
#%% 3d gif
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
def update(frame):
    ax.view_init(elev=25, azim=frame)  # Update the view angle for each frame
    return scatter,
animation = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=150)
animation.save('3D_plot.gif', writer='pillow')

#%% do cluster for embedding data.
from sklearn.cluster import DBSCAN
from sklearn import metrics
embeddings = reducer3d.embedding_
db = DBSCAN(eps=0.45, min_samples=10).fit(embeddings)
clust_labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(clust_labels)) - (1 if -1 in clust_labels else 0)
n_noise_ = list(clust_labels).count(-1)
print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
#%% with  3D plot
plt.clf()
plt.switch_backend('webAgg')
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
ax.grid(False)
# ax.axis('off')
unique_labels = np.unique(clust_labels)
handles = []
all_scatters = []
for label in unique_labels:
    mask = clust_labels == label
    scatter = ax.scatter3D(embeddings[:,0][mask], embeddings[:,1][mask], embeddings[:,2][mask], label=label,s = 15)
    all_scatters.append(scatter)
    handles.append(scatter)
ax.legend(handles=handles)
plt.show()
#%% get specific stimuli in given clust.
specific_clust = 0
clust_id = np.where(clust_labels == specific_clust)[0]
# get random 9 samples of given clust.
rand_clust_samples = random.sample(list(clust_id),9)
# plot 9 graphs in this clust.
plt.switch_backend('webAgg')
fig,ax = plt.subplots(3,3,figsize = (8,8))
fig.suptitle(f"Class{specific_clust}", fontsize=16)
for i,c_sample in enumerate(rand_clust_samples):
    c_path = all_graph_name[c_sample]
    c_graph = cv2.imread(c_path,cv2.COLOR_BGR2RGB)
    ax[i//3,i%3].imshow(c_graph)
    # ax[i//3,i%3].set_title(f'Class {i}')
    ax[i//3,i%3].axis('off')
plt.show()
# print original statistics in clust ids.
real_ids = labels[clust_id]
clust_num = len(clust_id)
print(f'Cluster {specific_clust} have cells as below: \n')
print(f'Orientation Count{np.sum(real_ids == 1)}/{clust_num}')
print(f'Color Count{np.sum(real_ids == 2)}/{clust_num}')
print(f'Curvature Count{np.sum(real_ids == 3)}/{clust_num}')
print(f'Non-domain Count{np.sum(real_ids == 4)}/{clust_num}')
#%% Average graphs in each cluster. 0-8 in each subplots.
all_avr_graph = np.zeros(shape = (300,300,3,9),dtype = 'u1')
for i in range(9):
    clust_id = np.where(clust_labels == i-1)[0]
    frame_num = len(clust_id)
    all_graphs = np.zeros(shape = (300,300,3,frame_num),dtype = 'f8')
    for j,c_sample in enumerate(clust_id):
        c_path = all_graph_name[c_sample]
        c_graph = cv2.imread(c_path,cv2.COLOR_BGR2RGB)
        all_graphs[:,:,:,j] = c_graph
    all_avr_graph[:,:,:,i] = np.clip(all_graphs.mean(3),0,255).astype('u1')
#%% Plot averaged graph.
plt.switch_backend('webAgg')
fig,ax = plt.subplots(3,3,figsize = (8,8))
fig.suptitle(f"All Averaged graph", fontsize=16)
for i in range(9):
    ax[i//3,i%3].imshow(all_avr_graph[:,:,:,i])
    ax[i//3,i%3].set_title(f'Class {i-1}')
    ax[i//3,i%3].axis('off')
plt.show()