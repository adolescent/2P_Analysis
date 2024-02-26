'''
Label OD data on manifold.
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
acd = ot.Load_Variable(wp,'All_Series_Dic91.pkl')
#%% Get OD data with labels.
labeled_data_dict = dict(tuple(labeled_data.groupby('From_Stim')))
od_frame = labeled_data_dict['OD'].reset_index(drop = True)
g16_frame = labeled_data_dict['G16'].reset_index(drop = True)
hue_frame = labeled_data_dict['Hue7Orien4'].reset_index(drop = True)
od_framenum = od_frame.shape[0]
g16_framenum = g16_frame.shape[0]
od_datasets = np.zeros(shape = (od_framenum,cell_num),dtype = 'f8')
od_labels = []
for i in range(od_framenum):
    c_frame = np.array(od_frame.loc[i,'Data'])
    od_datasets[i,:] = c_frame
    od_labels.append(od_frame.loc[i,'OD_Label_Num'])
g16_datasets = np.zeros(shape = (g16_framenum,cell_num),dtype = 'f8')
g16_labels = []
for i in range(g16_framenum):
    c_frame = np.array(g16_frame.loc[i,'Data'])
    g16_datasets[i,:] = c_frame
    g16_labels.append(g16_frame.loc[i,'OD_Label_Num'])

#%% Train an umap for od datas.
reducer = umap.UMAP(n_neighbors = 20,min_dist=0.01,n_components=2)
reducer.fit(od_datasets) # supervised learning on G16 data.
# reducer.fit(g16_datasets) # unsupervised learning on G16 data.
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
umap.plot.points(reducer,ax = ax,labels = np.array(od_labels),theme = 'fire')
plt.show()
ot.Save_Variable(wp,'OD_Model_All_unsupervised',reducer)

#%% Supervised OD. Actually used on clustering.
reducer2 = umap.UMAP(n_neighbors = 20,min_dist=0.01,n_components=2,target_weight= 0.03)
reducer2.fit(od_datasets,od_labels) # supervised learning on G16 data.
# reducer.fit(g16_datasets) # unsupervised learning on G16 data.

plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
umap.plot.points(reducer2,ax = ax,labels = np.array(od_labels),theme = 'fire')
plt.show()
ot.Save_Variable(wp,'OD_Model_All_supervised_20',reducer2)

#%% embedding G16 both-eye data on OD umap spaces.
g16_embeddings = reducer.transform(g16_datasets)
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
# umap.plot.points(reducer,ax = ax,labels = np.array(od_labels),theme = 'fire')
# ax.set_facecolor("black")
sns.kdeplot(x = reducer.embedding_[:,0],y = reducer.embedding_[:,1],hue = od_labels,hue_norm=(-1,3),ax = ax,palette = 'rainbow',levels = 5,alpha = 0.3)
sns.kdeplot(x = g16_embeddings[:,0],y = g16_embeddings[:,1],hue = g16_labels,hue_norm=(-1,3),ax = ax,palette = 'rainbow',levels = 5,alpha = 0.3)
sns.scatterplot(x = reducer.embedding_[:,0],y = reducer.embedding_[:,1],s = 16, marker='^',hue = od_labels,hue_norm=(-1,3),ax = ax,palette = 'gnuplot')
sns.scatterplot(x = g16_embeddings[:,0],y = g16_embeddings[:,1],s = 10,hue = g16_labels,hue_norm=(-1,3),ax = ax,palette = 'gnuplot')
plt.show()
#%% embedding spon data on OD umaps.
spon_embeddings = reducer.transform(np.array(spon_data))
# plt.clf()
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
umap.plot.points(reducer,ax = ax,labels = np.array(od_labels),theme = 'fire')
sns.scatterplot(x = spon_embeddings[:,0],y = spon_embeddings[:,1],ax = ax,s = 3)
sns.kdeplot(x = spon_embeddings[:,0],y = spon_embeddings[:,1],ax = ax,levels = 20)
plt.show()
#%% for supervised learning
spon_embeddings = reducer2.transform(np.array(spon_data))
# g16_embeddings = reducer2.transform(g16_datasets)
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
umap.plot.points(reducer2,ax = ax,labels = np.array(od_labels),theme = 'fire')
# sns.scatterplot(x = g16_embeddings[:,0],y = g16_embeddings[:,1],ax = ax,s = 3)
sns.scatterplot(x = spon_embeddings[:,0],y = spon_embeddings[:,1],ax = ax,s = 3,color = 'w')
# sns.kdeplot(x = spon_embeddings[:,0],y = spon_embeddings[:,1],ax = ax,levels = 20)
plt.show()

#%% SVC on unsupervised learning.
from sklearn import svm
from sklearn.model_selection import cross_val_score
od_embed_unsup = reducer.embedding_
od_embed_sup = reducer2.embedding_
model = svm.SVC()
scores = cross_val_score(model, od_embed_unsup, od_labels, cv=5)
print(f'Score of 5 fold SVC on OD unsupervised : {scores.mean()*100:.2f}%')
#%% train a real one.
od_model = svm.SVC(kernel = 'rbf',probability=True)
od_predict = od_model.fit(od_embed_sup,od_labels)
spon_embedding_sup = reducer2.transform(spon_data)
spon_probability = od_model.predict_proba(spon_embedding_sup)

#%% label spon with given prob.
prob_thres = 0.99
predicted_spon_labels = np.zeros(spon_probability.shape[0])
for i in range(spon_probability.shape[0]):
    c_prob = spon_probability[i,:]
    c_max = c_prob.max()
    if c_max<prob_thres:
        predicted_spon_labels[i] =-1
    else:
        predicted_spon_labels[i] = np.where(c_prob == c_max)[0][0]
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (10,10))
ax.set_facecolor("black")
ax = sns.scatterplot(x = spon_embedding_sup[:,0],y = spon_embedding_sup[:,1],s = 2,hue = predicted_spon_labels,legend = 'full',palette = 'brg')
plt.show()
#%% Average OD graphs
from Cell_Tools.Cell_Visualization import Cell_Weight_Visualization
stim_graphs = np.zeros(shape = (3,651),dtype = 'f8')
visualized_all_graph = np.zeros(shape = (512,512,3))
for i in range(3):
    c_id_frames = np.where(predicted_spon_labels == i)[0]
    c_frames = spon_data.loc[c_id_frames,:]
    stim_graphs[i] = c_frames.mean(0)
    visualized_all_graph[:,:,i] = Cell_Weight_Visualization(np.array(c_frames.mean(0)),acd)

plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,3,figsize = (15,15))
for i in range(3):
    sns.heatmap(visualized_all_graph[:,:,i],ax = ax[i],center = 0,square = True,xticklabels=False, yticklabels=False,cbar = False)
    ax[i].set_title(f'Class {i}')
plt.show()