'''

This script train an svc on given data, do k-fold and prediction.

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


wp = r'D:\ZR\_Data_Temp\_Temp_Data\220711_temp\UMAP_Datas'
condapath = r'C:\ProgramData\anaconda3\envs\umapzr'
kill_all_cache(condapath)
labeled_data = ot.Load_Variable(wp,'Frame_ID_infos.pkl')
spon_data = ot.Load_Variable(wp,'Z_mean_Run01.pkl')
cell_num = labeled_data.iloc[0,0].shape[0]
#%% Labeled data dict 
labeled_data_dict = dict(tuple(labeled_data.groupby('From_Stim')))
#%% get each functional datas.
od_frame = labeled_data_dict['OD'].reset_index(drop = True)
g16_frame = labeled_data_dict['G16'].reset_index(drop = True)
hue_frame = labeled_data_dict['Hue7Orien4'].reset_index(drop = True)

#%% Get orientation embedding from g16 frames.
g16_framenum = g16_frame.shape[0]
g16_datasets = np.zeros(shape = (g16_framenum,cell_num),dtype = 'f8')
g16_labels = []
for i in range(g16_framenum):
    c_frame = np.array(g16_frame.loc[i,'Data'])
    g16_datasets[i,:] = c_frame
    g16_labels.append(g16_frame.loc[i,'Orien_Label_Num'])
#%% UMAP
reducer = umap.UMAP(n_neighbors = 20,min_dist=0.01,n_components=2)
reducer.fit(g16_datasets,g16_labels) # supervised learning on G16 data.
# reducer.fit(g16_datasets) # unsupervised learning on G16 data.
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
umap.plot.points(reducer,ax = ax,labels = np.array(g16_labels),theme = 'fire')
plt.show()
ot.Save_Variable(wp,'G16_Model_All_supervised',reducer)
#%% UMAP on no-ISI data.
only_stim_ids = np.where(np.array(g16_labels) != 0)[0]
g16_stim_labels = np.array(g16_labels)[only_stim_ids]
g16_stim_datasets = g16_datasets[only_stim_ids,:]
reducer2 = umap.UMAP(n_neighbors = 20,min_dist=0.01,n_components=2)
reducer2.fit(g16_stim_datasets,g16_stim_labels) # supervised learning on G16 data.
# reducer2.fit(only_stim_datasets) # unsupervised learning on G16 data.
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
umap.plot.points(reducer2,ax = ax,labels = g16_stim_labels,theme = 'fire')
plt.show()
ot.Save_Variable(wp,'G16_Model_Stimonly_supervised',reducer2)
#%% Do svc on G16 data.
from sklearn import svm
from sklearn.model_selection import cross_val_score
g16_embed_all = reducer.embedding_
g16_embed_stim = reducer2.embedding_
model = svm.SVC()
scores_all = cross_val_score(model, g16_embed_all, g16_labels, cv=5)
scores_stim = cross_val_score(model, g16_embed_stim, g16_stim_labels, cv=5)
print(f'Score of 5 fold SVC on G16 all:{scores_all.mean()*100:.2f}%;on G16 all:{scores_stim.mean()*100:.2f}%')
#%% embedding OD and Hue data into G16 trained umap space.
od_framenum = od_frame.shape[0]
od_datasets = np.zeros(shape = (od_framenum,cell_num),dtype = 'f8')
od_labels = np.zeros(od_framenum)
for i in range(od_framenum):
    c_frame = np.array(od_frame.loc[i,'Data'])
    od_datasets[i,:] = c_frame
    od_labels[i]=(od_frame.loc[i,'Orien_Label_Num'])
hue_framenum = hue_frame.shape[0]
hue_datasets = np.zeros(shape = (hue_framenum,cell_num),dtype = 'f8')
hue_labels = np.zeros(hue_framenum)
for i in range(hue_framenum):
    c_frame = np.array(hue_frame.loc[i,'Data'])
    hue_datasets[i,:] = c_frame
    hue_labels[i]=(hue_frame.loc[i,'Orien_Label_Num'])
#%% embedding OD data on graph.
OD_embedding = reducer.transform(od_datasets)
#%% get stim only OD embeddings 
only_stim_ids = np.where(od_labels != 0)[0]
od_stim_labels = od_labels[only_stim_ids]
od_stim_datasets = od_datasets[only_stim_ids,:]
OD_embedding_stim = reducer2.transform(od_stim_datasets)

plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (10,10))
ax.set_facecolor("black")
# ax = sns.scatterplot(x = raw_g16_embedding[:,0],y = raw_g16_embedding[:,1],s = 2,hue = np.array(g16_labels),palette = 'Spectral',legend = 'full')
# umap.plot.points(reducer,ax = ax,labels = np.array(g16_labels),theme = 'fire')
umap.plot.points(reducer2,ax = ax,labels = g16_stim_labels,theme = 'fire')
ax = sns.scatterplot(x = OD_embedding_stim[:,0],y = OD_embedding_stim[:,1],color = 'white',s = 3)
plt.show()
#%% Train a SVC through G16 data, and test preformance on OD data.
G16_SVC_all = svm.SVC(kernel='sigmoid',coef0=0.7)
G16_SVC_all.fit(X = reducer.embedding_,y = g16_labels)
pred_OD_label = G16_SVC_all.predict(OD_embedding)
pred_score = G16_SVC_all.score(OD_embedding,od_labels)
G16_SVC_stim = svm.SVC(kernel='linear',coef0=0.0)
G16_SVC_stim.fit(X = reducer2.embedding_,y = g16_stim_labels)
pred_OD_label_stim = G16_SVC_stim.predict(OD_embedding_stim)
pred_score_stim = G16_SVC_stim.score(OD_embedding_stim,od_stim_labels)

print(f'Score of all frame SVC:{pred_score*100:.2f}%;stim frame SVC:{pred_score_stim*100:.2f}%')
#%% Embedding spontaneous data into given manifold space.
spon_embedding_g16_all = reducer.transform(spon_data)
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (10,10))
umap.plot.points(reducer,ax = ax,labels = np.array(g16_labels),theme = 'fire')
ax = sns.scatterplot(x = spon_embedding_g16_all[:,0],y = spon_embedding_g16_all[:,1],color = 'white',s = 3)
ax = sns.kdeplot(x = spon_embedding_g16_all[:,0],y = spon_embedding_g16_all[:,1],fill=False, thresh=0.01, levels=10, color = 'white')
plt.show()
#%% Embedding on stim only manifold space.
spon_embedding_g16_stim = reducer2.transform(spon_data)
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (10,10))
umap.plot.points(reducer2,ax = ax,labels = np.array(g16_stim_labels),theme = 'fire')
ax = sns.scatterplot(x = spon_embedding_g16_stim[:,0],y = spon_embedding_g16_stim[:,1],color = 'white',s = 3)
ax = sns.kdeplot(x = spon_embedding_g16_stim[:,0],y = spon_embedding_g16_stim[:,1],fill=False, thresh=0.01, levels=10, color = 'white')
plt.show()
#%% Use the G16 SVC classify spontaneous data, and refuse specific cr-rate.
# G16_SVC_all = svm.SVC(kernel='sigmoid',coef0=0.2,probability=True)
G16_SVC_all = svm.SVC(kernel='rbf',probability=True)
G16_SVC_all.fit(X = reducer.embedding_,y = g16_labels)
spon_probability_all = G16_SVC_all.predict_proba(spon_embedding_g16_all)
prob_thres = 0.9
predicted_spon_labels = np.zeros(spon_probability_all.shape[0])
# label spon with given prob.
for i in range(spon_probability_all.shape[0]):
    c_prob = spon_probability_all[i,:]
    c_max = c_prob.max()
    if c_max<prob_thres:
        predicted_spon_labels[i] =-1
    else:
        predicted_spon_labels[i] = np.where(c_prob == c_max)[0][0]
# plot svc with label on spon data.
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (10,10))
ax.set_facecolor("black")
ax = sns.scatterplot(x = spon_embedding_g16_all[:,0],y = spon_embedding_g16_all[:,1],s = 2,hue = predicted_spon_labels,legend = 'full',palette = 'brg')
plt.show()
#%% Get average graph vs stim functional map.
from Cell_Tools.Cell_Visualization import Cell_Weight_Visualization
all_graphs = np.zeros(shape = (9,651),dtype = 'f8')
acd = ot.Load_Variable(wp,'All_Series_Dic91.pkl')
visualized_all_graph = np.zeros(shape = (512,512,9))
for i in range(9):
    c_id_frames = np.where(predicted_spon_labels == i)[0]
    c_frames = spon_data.loc[c_id_frames,:]
    all_graphs[i] = c_frames.mean(0)
    visualized_all_graph[:,:,i] = Cell_Weight_Visualization(np.array(c_frames.mean(0)),acd)

plt.switch_backend('webAgg')
fig,ax = plt.subplots(3,3,figsize = (15,15))
for i in range(9):
    sns.heatmap(visualized_all_graph[:,:,i],ax = ax[i//3,i%3],center = 0,square = True,xticklabels=False, yticklabels=False,cbar = False)
    ax[i//3,i%3].set_title(f'Class {i}')
plt.show()

#%% repeat upper operation on stim only embeddings.(reducer2)
G16_SVC_stim = svm.SVC(kernel='rbf',coef0=0.3,probability=True)
G16_SVC_stim.fit(X = reducer2.embedding_,y = g16_stim_labels)
spon_probability_stim = G16_SVC_stim.predict_proba(spon_embedding_g16_stim)
prob_thres = 0.925
predicted_spon_labels = np.zeros(spon_probability_stim.shape[0])
# label spon with given prob.
for i in range(spon_probability_stim.shape[0]):
    c_prob = spon_probability_stim[i,:]
    c_max = c_prob.max()
    if c_max<prob_thres:
        predicted_spon_labels[i] =-1
    else:
        predicted_spon_labels[i] = np.where(c_prob == c_max)[0][0]+1
# plot svc with label on spon data.
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (10,10))
ax.set_facecolor("black")
ax = sns.scatterplot(x = spon_embedding_g16_stim[:,0],y = spon_embedding_g16_stim[:,1],s = 2,hue = predicted_spon_labels,legend = 'full',palette = 'brg')
plt.show()
#%% The same as all data, plot is also possible.
stim_graphs = np.zeros(shape = (8,651),dtype = 'f8')
visualized_all_graph = np.zeros(shape = (512,512,8))
for i in range(8):
    c_id_frames = np.where(predicted_spon_labels == i+1)[0]
    c_frames = spon_data.loc[c_id_frames,:]
    stim_graphs[i] = c_frames.mean(0)
    visualized_all_graph[:,:,i] = Cell_Weight_Visualization(np.array(c_frames.mean(0)),acd)

plt.switch_backend('webAgg')
fig,ax = plt.subplots(3,3,figsize = (15,15))
for i in range(8):
    sns.heatmap(visualized_all_graph[:,:,i],ax = ax[(i+1)//3,(i+1)%3],center = 0,square = True,xticklabels=False, yticklabels=False,cbar = False)
    ax[(i+1)//3,(i+1)%3].set_title(f'Class {i+1}')
plt.show()
#%% Direct connection map of G16 data, OD embeddings and spon embeddings.
# ORDER IS IMPORTANT!
# g16_2d = reducer.transform(orien_frames)
spon_2d = reducer.transform(spon_data)
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (10,10))
ax.set_facecolor("black")
ax.plot(spon_2d[:,0],spon_2d[:,1],color = 'y',alpha = 0.9)
# umap.plot.points(reducer,ax = ax,labels = np.array(g16_labels),theme = 'fire')
plt.show()