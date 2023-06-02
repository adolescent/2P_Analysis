'''
This script do umap on stim and isi. All Stim data is proveded, and color not included.

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


wp = r'D:\ZR\_Data_Temp\_Temp_Data\220711_temp\UMAP_Datas'
condapath = r'C:\ProgramData\anaconda3\envs\umapzr'
kill_all_cache(condapath)
labeled_data = ot.Load_Variable(wp,'Frame_ID_infos.pkl')
spon_data = ot.Load_Variable(wp,'Z_mean_Run01.pkl')
cell_num = labeled_data.iloc[0,0].shape[0]
acd = ot.Load_Variable(wp,'All_Series_Dic91.pkl')
# data frame with no color.
labeled_data_no_color = labeled_data[labeled_data['From_Stim']!= 'Hue7Orien4']
#%% If load in, run this.
reducer = ot.Load_Variable(wp,'Stim_With_ISI_UMAP_Unsup_3d.pkl')
u = reducer.embedding_
#%% Generate data with label.
frame_num = labeled_data_no_color.shape[0]
all_stim_frame = np.zeros(shape = (frame_num,cell_num),dtype = 'f8')
all_labels = []
eye_labels = []
orien_labels = []

for i in range(frame_num):
    c_slide = labeled_data_no_color.loc[i,:]
    all_stim_frame[i,:] = np.array(c_slide['Data'])
    eye_labels.append(c_slide['OD_Label_Num'])
    orien_labels.append(c_slide['Orien_Label_Num'])
    # calculate global id.
    if (c_slide['OD_Label_Num'] == 0) or (c_slide['Orien_Label_Num'] == 0):
        whole_id = 0
    else:
        whole_id = 8*(c_slide['OD_Label_Num']-1)+c_slide['Orien_Label_Num']
    all_labels.append(whole_id)
'''
After label generation, we get all eye, orien, both label.
1/3/5/7 as LE, 9/11/13/15 as RE, 17-24 as both eye.
'''    
#%% UMAP all data with ISI on umap space.
reducer = umap.UMAP(n_neighbors = 20,min_dist=0.01,n_components=3)
reducer.fit(all_stim_frame) # supervised learning on G16 data.
# reducer.fit(g16_datasets) # unsupervised learning on G16 data.
u = reducer.embedding_
ot.Save_Variable(wp,'Stim_With_ISI_UMAP_Unsup_3d',reducer)
#%% Embed spon data on this space.
spon_embeddings = reducer.transform(spon_data)
#%% Plot 3D graph.
plt.clf()
plt.switch_backend('webAgg')
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
ax.grid(False)
# ax.axis('off')
unique_labels = np.unique(orien_labels)
handles = []
all_scatters = []
for label in unique_labels:
    mask = orien_labels == label
    scatter = ax.scatter3D(u[:,0][mask], u[:,1][mask], u[:,2][mask], label=label,s = 5,alpha = 0.6)
    all_scatters.append(scatter)
    handles.append(scatter)
ax.scatter3D(spon_embeddings[:,0],spon_embeddings[:,1],spon_embeddings[:,2],s = 3,c = 'r')
# ax.scatter3D(shuffle_embeddings[:,0],shuffle_embeddings[:,1],shuffle_embeddings[:,2],s = 3,c = 'black')
ax.legend(handles=handles)
plt.show()
#%%
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
x = u[:,0]
y = u[:,1]
z = u[:,2]
# n = 2363
def update(frame):
    ax.view_init(elev=frame, azim=frame)  # Update the view angle for each frame
    return scatter,
animation = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=150)
animation.save('3D_plot.gif', writer='pillow')
#%% Do SVC on all label.
# Test k fold svc accuracy.
model = svm.SVC(C = 1)
scores_all = cross_val_score(model, u , all_labels, cv=10)
print(f'Score of 10 fold SVC on 3D all:{scores_all.mean()*100:.2f}%')
#%% Use SVC above do cluster on spon embeddings.
model = svm.SVC(C = 10,probability=True)
model.fit(X = u,y = all_labels)
spon_probability_all = model.predict_proba(spon_embeddings)
prob_thres = 0.5
predicted_spon_labels = np.zeros(spon_embeddings.shape[0])
raw_predicted_label = model.predict(spon_embeddings)
# label spon with given prob.
for i in range(spon_probability_all.shape[0]):
    c_prob = spon_probability_all[i,:]
    c_max = c_prob.max()
    if c_max<prob_thres:
        predicted_spon_labels[i] =-1
    else:
        # predicted_spon_labels[i] = np.where(c_prob == c_max)[0][0]
        predicted_spon_labels[i] = raw_predicted_label[i]
print(sum(predicted_spon_labels == -1))
# Plot 3D graph.
import matplotlib.cm as cm
colors = cm.turbo(np.linspace(0, 1, 17))# colorbars.
plt.clf()
plt.switch_backend('webAgg')
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
ax.grid(False)
# ax.axis('off')
unique_labels = np.unique(predicted_spon_labels)
handles = []
all_scatters = []
i = 0
for label in unique_labels:
    mask = predicted_spon_labels == label
    scatter = ax.scatter3D(spon_embeddings[:,0][mask], spon_embeddings[:,1][mask], spon_embeddings[:,2][mask], label=label,s = 5,alpha = 1,color = colors[i])
    all_scatters.append(scatter)
    handles.append(scatter)
    i += 1
ax.legend(handles=handles,ncol = 2)
plt.show()
#%% Average all svc averaged label graphs.
clust_sets = list(set(predicted_spon_labels))
clust_avr_graphs = np.zeros(shape = (len(clust_sets),cell_num),dtype = 'f8')
visual_graphs = np.zeros(shape = (512,512,len(clust_sets)),dtype = 'f8')
for i,c_clust in enumerate(clust_sets):
    c_loc = np.where(predicted_spon_labels == c_clust)[0]
    c_frames = spon_data.loc[c_loc].mean(0)
    clust_avr_graphs[i] = np.array(c_frames)
    visual_graphs[:,:,i] = Cell_Weight_Visualization(c_frames,acd)
#%% Plot all graphs in single subplots.
plt.switch_backend('webAgg')
fig,ax = plt.subplots(3,5,figsize = (12,8))
fig.suptitle(f"All Averaged graph", fontsize=16)
for i in range(15):
    sns.heatmap(visual_graphs[:,:,i],center = 0,square = True, xticklabels= False, yticklabels=False,ax = ax[i//5,i%5],cbar = False)
    ax[i//5,i%5].set_title(f'Class {clust_sets[i]}')
plt.show()
#%% Do unsupervised cluster.
from sklearn.cluster import DBSCAN
from sklearn import metrics
db = DBSCAN(eps=0.2, min_samples=10).fit(spon_embeddings)
clust_labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(clust_labels)) - (1 if -1 in clust_labels else 0)
n_noise_ = list(clust_labels).count(-1)
print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
#%% with  3D plot
import matplotlib.cm as cm
colors = cm.turbo(np.linspace(0, 1, n_clusters_+1))# colorbars.
plt.clf()
plt.switch_backend('webAgg')
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
ax.grid(False)
# ax.axis('off')
unique_labels = np.unique(clust_labels)
handles = []
all_scatters = []
i = 0
for label in unique_labels:
    mask = clust_labels == label
    scatter = ax.scatter3D(spon_embeddings[:,0][mask], spon_embeddings[:,1][mask], spon_embeddings[:,2][mask], label=label,s = 5,alpha = 1,color = colors[i])
    all_scatters.append(scatter)
    handles.append(scatter)
    i += 1
ax.legend(handles=handles,ncol = 2)
plt.show()
#%% Plot 3D Clusts in single frame.
clust_sets = list(set(clust_labels))
clust_avr_graphs = np.zeros(shape = (len(clust_sets),cell_num),dtype = 'f8')
visual_graphs = np.zeros(shape = (512,512,len(clust_sets)),dtype = 'f8')
for i,c_clust in enumerate(clust_sets):
    c_loc = np.where(clust_labels == c_clust)[0]
    c_frames = spon_data.loc[c_loc].mean(0)
    clust_avr_graphs[i] = np.array(c_frames)
    visual_graphs[:,:,i] = Cell_Weight_Visualization(c_frames,acd)
plt.switch_backend('webAgg')
fig,ax = plt.subplots(3,5,figsize = (12,8))
fig.suptitle(f"All Averaged graph", fontsize=16)
for i in range(15):
    sns.heatmap(visual_graphs[:,:,i],center = 0,square = True, xticklabels= False, yticklabels=False,ax = ax[i//5,i%5],cbar = False)
    ax[i//5,i%5].set_title(f'Class {clust_sets[i]}')
plt.show()

#%% Last, shuffle each cell train, let's see how this will embedding in umap.
shuffled_spondata = np.zeros(shape = spon_data.shape,dtype = 'f8')
for i in range(cell_num):
    cc_series = spon_data.iloc[:,i].sample(frac = 1).reset_index(drop = True)
    shuffled_spondata[:,i] = cc_series
shuffle_embeddings = reducer.transform(shuffled_spondata)

