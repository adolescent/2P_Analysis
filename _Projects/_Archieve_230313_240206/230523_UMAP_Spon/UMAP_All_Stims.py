'''
This script generate all stim umap.

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
# data frame with no isi.
labeled_data_no_isi = labeled_data[labeled_data['Raw_ID']!= '-1']
labeled_data_no_isi = labeled_data_no_isi[labeled_data_no_isi['Color_Label']!= 'White'].reset_index(drop = True)
#%% If load in, run this.
reducer = ot.Load_Variable(wp,'Stim_No_ISI_UMAP_Unsup_3d.pkl')
u = reducer.embedding_
#%% Get Label ID with np data frame.
frame_num = labeled_data_no_isi.shape[0]
all_stim_frame = np.zeros(shape = (frame_num,cell_num),dtype = 'f8')
all_labels = []
eye_labels = []
orien_labels = []
color_labels = []
for i in range(frame_num):
    c_slide = labeled_data_no_isi.loc[i,:]
    all_stim_frame[i,:] = np.array(c_slide['Data'])
    eye_labels.append(c_slide['OD_Label_Num'])
    orien_labels.append(c_slide['Orien_Label_Num'])
    color_labels.append(c_slide['Color_Label_Num'])
    if c_slide['Color_Label_Num'] != 0:# making color16-19,and 1-4LE,5-8RE,9-16Orien
        c_id = 24+c_slide['Color_Label_Num']
    else:
        c_id = 8*(c_slide['OD_Label_Num']-1)+c_slide['Orien_Label_Num']
    all_labels.append(c_id)
#%% UMAP them.
reducer = umap.UMAP(n_neighbors = 20,min_dist=0.01,n_components=3)
reducer.fit(all_stim_frame) # supervised learning on G16 data.
# reducer.fit(g16_datasets) # unsupervised learning on G16 data.
u = reducer.embedding_
ot.Save_Variable(wp,'Stim_No_ISI_UMAP_Unsup_3d',reducer)
# plt.switch_backend('webAgg')
# fig,ax = plt.subplots(1,figsize = (12,12))
# umap.plot.points(reducer,ax = ax,labels = np.array(all_labels),theme = 'fire')
# plt.show()
#%%
plt.clf()
plt.switch_backend('webAgg')
# fig,ax = plt.subplots(1,figsize = (12,12))
# umap.plot.points(reducer,ax = ax,labels = np.array(all_labels),theme = 'inferno')
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
ax.grid(False)
# ax.axis('off')
unique_labels = np.unique(eye_labels)
handles = []
# scatter = ax.scatter3D(u[:,0],u[:,1],u[:,2], c=orien_labels,label = orien_labels, cmap='jet', s=3)
all_scatters = []
for label in unique_labels:
    mask = eye_labels == label
    scatter = ax.scatter3D(u[:,0][mask], u[:,1][mask], u[:,2][mask], label=label,s = 5)
    all_scatters.append(scatter)
    handles.append(scatter)
ax.legend(handles=handles)
# ax.set_xlim(2, 8)  # Set X-axis range
# ax.set_ylim(8, 15)  # Set Y-axis range
# ax.set_zlim(3, 11)  # Set Z-axis range
# ax.view_init(elev=30, azim=0)
plt.show()
#%% Save current graph as gif
# Warnings! Using this animation will lead to bug in graph show, do this after all operations on 2D plots.
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
#%% embedding isi in stim space.
labeled_data_isi = labeled_data[labeled_data['Raw_ID']== '-1'].reset_index(drop = True)
all_isi_frame = np.zeros(shape = (labeled_data_isi.shape[0],cell_num),dtype = 'f8')
for i in range(labeled_data_isi.shape[0]):
    all_isi_frame[i,:] = np.array(labeled_data_isi.loc[i,'Data'])
# embed data on 3d reducer.
isi_embedding = reducer.transform(all_isi_frame)
#%% Plot data on umap space.
plt.switch_backend('webAgg')
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
ax.grid(False)
# ax.axis('off')
unique_labels = np.unique(orien_labels)
handles = []
for label in unique_labels:
    mask = orien_labels == label
    scatter = ax.scatter3D(u[:,0][mask], u[:,1][mask], u[:,2][mask], label=label,s = 3,alpha = 0.5)
    handles.append(scatter)
ax.scatter3D(isi_embedding[:,0],isi_embedding[:,1],isi_embedding[:,2],s = 5)
# ax.scatter3D(shuffle_embeddings[:,0],shuffle_embeddings[:,1],shuffle_embeddings[:,2],s = 3,c = 'black')
ax.legend(handles=handles)
plt.show()
#%% Embed spon data on manifold above.
spon_embedding = reducer.transform(spon_data)

#%% for a consequent time, embedding dynamical trajectory of cell reaction.
acd = ot.Load_Variable(wp,'All_Series_Dic91.pkl')
from Filters import Signal_Filter
def Given_Run_Frame(acd,runname = '1-001',fps = 1.301):
    acn = list(acd.keys())
    frame_num = len(acd[1][runname])
    selected_frames = pd.DataFrame(columns = acn,index= list(range(frame_num)))
    for i,cc in enumerate(acn):
        cc_run = acd[cc][runname]
        filted_cc_run = Signal_Filter(cc_run,order =7,filter_para = (0.01/fps,0.6/fps))
        selected_frames[cc] = filted_cc_run
    return selected_frames
g16_frames = Given_Run_Frame(acd,'1-007')
g16_frames = (g16_frames-g16_frames.mean())/g16_frames.mean()
g16_frames = g16_frames/g16_frames.std()
# embed all g16 on origin reducer.
g16_embeddings = reducer.transform(g16_frames)
#%% plot g16 on original 3d scatters.
used_embeddings = g16_embeddings[500:1000,:]
plt.switch_backend('webAgg')
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
ax.grid(False)

# scatter = ax.scatter3D(u[:,0],u[:,1],u[:,2],c = 'blue')

def update(frame):
    # Update the position of the scatter plot
    ax.cla()
    # scatter._offsets3d = (u[:,0],u[:,1],u[:,2])
    scatter = ax.scatter3D(u[:,0],u[:,1],u[:,2],c = 'blue',s = 5,alpha = 0.4)
    # Update the position of the trajectory point
    ax.scatter3D(used_embeddings[:,0][frame], used_embeddings[:,1][frame], used_embeddings[:,2][frame], c='red', s = 50)
    return scatter,

# Create the animation
animation = FuncAnimation(fig, update, frames=200, interval=730)
animation.save('3D_plot.mp4', writer='ffmpeg')
# Show the plot
plt.show()
#%% Do SVC on Stim only manifold.
model = svm.SVC(C = 10,probability=True)
model.fit(X = u,y = all_labels)
spon_probability_all = model.predict_proba(spon_embedding)
prob_thres = 0.5
predicted_spon_labels = np.zeros(spon_embedding.shape[0])
raw_predicted_label = model.predict(spon_embedding)
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
colors = cm.turbo(np.linspace(0, 1, 18))# colorbars.
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
    scatter = ax.scatter3D(spon_embedding[:,0][mask], spon_embedding[:,1][mask], spon_embedding[:,2][mask], label=label,s = 5,alpha = 1,color = colors[i])
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
fig,ax = plt.subplots(3,6,figsize = (15,8))
fig.suptitle(f"SVC Averaged graph", fontsize=16)
for i in range(18):
    sns.heatmap(visual_graphs[:,:,i],center = 0,square = True, xticklabels= False, yticklabels=False,ax = ax[i//6,i%6],cbar = False)
    ax[i//6,i%6].set_title(f'Class {clust_sets[i]}')
plt.show()
#%% Unsupervised DBSCAN
from sklearn.cluster import DBSCAN
from sklearn import metrics
db = DBSCAN(eps=0.2, min_samples=10).fit(spon_embedding)
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
    scatter = ax.scatter3D(spon_embedding[:,0][mask], spon_embedding[:,1][mask], spon_embedding[:,2][mask], label=label,s = 5,alpha = 1,color = colors[i])
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
fig,ax = plt.subplots(3,7,figsize = (16,8))
fig.suptitle(f"All Averaged graph", fontsize=16)
for i in range(21):
    sns.heatmap(visual_graphs[:,:,i],center = 0,square = True, xticklabels= False, yticklabels=False,ax = ax[i//7,i%7],cbar = False)
    ax[i//7,i%7].set_title(f'Class {clust_sets[i]}')
plt.show()