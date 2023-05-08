'''
After generating a data label, here we try manifold on given data.

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

#%%
wp = r'D:\ZR\_Data_Temp\_Temp_Data\220711_temp\UMAP_Datas'
labeled_data = ot.Load_Variable(wp,'Frame_ID_infos.pkl')
dff_data = ot.Load_Variable(wp,'Dff_mean_Run01.pkl')
#%% encode label data of orientation only.
total_frame_num = labeled_data.shape[0]
cell_num = dff_data.shape[1]
all_labels = np.zeros(total_frame_num,dtype = 'f8')
all_frames = np.zeros(shape = (total_frame_num,cell_num),dtype = 'f8')
for i in tqdm(range(total_frame_num)):
    c_frame = np.array(labeled_data.iloc[i,0])
    all_frames[i,:] = c_frame
    raw_orien_label = labeled_data.iloc[i,4]
    if raw_orien_label == 'Orien0':
        all_labels[i] = 1
    elif raw_orien_label == 'Orien45':
        all_labels[i] = 2
    elif raw_orien_label == 'Orien90':
        all_labels[i] = 3
    elif raw_orien_label == 'Orien135':
        all_labels[i] = 4
    else:# for non-orientation frames.
        all_labels[i] = 5
    
#%% train umap for given G16 data.
reducer = umap.UMAP(n_neighbors = 10,min_dist=0.01,n_components=2)
reducer.fit(all_frames,all_labels)
#%% plot trained umap data.
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
umap.plot.points(reducer,ax = ax,labels = all_labels,theme = 'fire')
plt.show()
#%% plot connectivity
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (10,10))
umap.plot.connectivity(reducer,show_points= True)
# umap.plot.connectivity(reducer,edge_bundling='hammer')
plt.show()
#%% fit trained umap on spon data.
reduced_spon_data = reducer.transform(np.array(dff_data))
# reduced_spon_data = reducer.transform(np.array(g16_frames))
# plot reduced spon data on frame.
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (10,10))
ax = plt.scatter(reduced_spon_data[:,0],reduced_spon_data[:,1],s = 3)
plt.show()
#%% process pca reduced data.
# from Series_Analyzer.Cell_Frame_PCA_Cai import PC_Reduction
# pc_reducted_series = PC_Reduction(dff_data,PC_Range = [2,100])
# reduced_spon_data = reducer.transform(np.array(pc_reducted_series))
# # plot reduced spon data on frame.
# plt.switch_backend('webAgg')
# fig,ax = plt.subplots(1,figsize = (10,10))
# ax = plt.scatter(reduced_spon_data[:,0],reduced_spon_data[:,1],s = 3)
# plt.show()
#%% Do DBSCAN cluster on given data.
from sklearn.cluster import DBSCAN
from sklearn import metrics
db = DBSCAN(eps=0.6, min_samples=10).fit(reduced_spon_data)
labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (10,10))
ax = sns.scatterplot(x = reduced_spon_data[:,0],y = reduced_spon_data[:,1],s = 10,hue = labels,palette = 'tab10')
plt.show()
#%% get specific labels.
label2_ids = np.where(labels == 7)[0]
label2_frames = np.array(dff_data)[label2_ids,:]
mean_graph = label2_frames.mean(0)
# show each groups.
acd = ot.Load_Variable(wp,'All_Series_Dic91.pkl')
def Cell_weight_visualization(weights,acd):
    visualized_graph = np.zeros(shape = (512,512),dtype = 'f8')
    for i,c_weight in enumerate(weights):
        cc_x,cc_y = acd[i+1]['Cell_Loc']
        cc_loc = (acd[i+1]['Cell_Loc'].astype('i4')[1],acd[i+1]['Cell_Loc'].astype('i4')[0])
        visualized_graph = cv2.circle(visualized_graph,cc_loc,4,c_weight,-1)
    return visualized_graph

label2_visual = Cell_weight_visualization(mean_graph,acd)
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (10,10))
ax = sns.heatmap(label2_visual,center = 0, xticklabels=False, yticklabels=False,square = True)
plt.show()
