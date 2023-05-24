'''
This project is used on OD dimension reduction. we mix OD and G16 to generate a 4 class classifier, both eye information added.
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
#%% generate data frame.
labeled_data_dict = dict(tuple(labeled_data.groupby('From_Stim')))
od_frame = labeled_data_dict['OD'].reset_index(drop = True)
g16_frame = labeled_data_dict['G16'].reset_index(drop = True)
mix_frame = pd.concat([od_frame,g16_frame]).reset_index(drop=True)
mix_framenum = mix_frame.shape[0]
mix_datasets = np.zeros(shape = (mix_framenum,cell_num),dtype = 'f8')
mix_labels = []
for i in range(mix_framenum):
    c_frame = np.array(mix_frame.loc[i,'Data'])
    mix_datasets[i,:] = c_frame
    mix_labels.append(mix_frame.loc[i,'OD_Label_Num'])
#%% Let's UMAP them!
reducer = umap.UMAP(n_neighbors = 20,min_dist=0.01,n_components=2)
reducer.fit(mix_datasets) # supervised learning on G16 data.
# reducer.fit(g16_datasets) # unsupervised learning on G16 data.
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
umap.plot.points(reducer,ax = ax,labels = np.array(mix_labels),theme = 'fire')
plt.show()
ot.Save_Variable(wp,'OD_Model_Botheye_unsupervised',reducer)

#%% Supervised UMAP
reducer2 = umap.UMAP(n_neighbors = 30,min_dist=0.01,n_components=2,target_weight= 0.05)
reducer2.fit(mix_datasets,mix_labels) # supervised learning on G16 data.
# reducer.fit(g16_datasets) # unsupervised learning on G16 data.
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
umap.plot.points(reducer2,ax = ax,labels = np.array(mix_labels),theme = 'fire')
plt.show()
ot.Save_Variable(wp,'OD_Model_Botheye_supervised',reducer2)
#%% embedding spontaneous on supervised reducer.
spon_embeddings = reducer2.transform(spon_data)
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
umap.plot.points(reducer2,ax = ax,labels = np.array(mix_labels),theme = 'fire')
sns.scatterplot(x = spon_embeddings[:,0],y = spon_embeddings[:,1],ax = ax,s = 3,color = 'w')
plt.show()
#%% train SVC
model = svm.SVC(kernel = 'rbf',probability=True)
predict = model.fit(reducer2.embedding_,mix_labels)
spon_probability = model.predict_proba(spon_embeddings)
#%% label spon with given prob.
prob_thres = 0.97
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
ax = sns.scatterplot(x = spon_embeddings[:,0],y = spon_embeddings[:,1],s = 2,hue = predicted_spon_labels,legend = 'full',palette = 'brg')
plt.show()
#%% visualize.
stim_graphs = np.zeros(shape = (4,651),dtype = 'f8')
visualized_all_graph = np.zeros(shape = (512,512,4))
for i in range(4):
    c_id_frames = np.where(predicted_spon_labels == i)[0]
    c_frames = spon_data.loc[c_id_frames,:]
    stim_graphs[i] = c_frames.mean(0)
    visualized_all_graph[:,:,i] = Cell_Weight_Visualization(np.array(c_frames.mean(0)),acd)

plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,4,figsize = (15,15))
for i in range(4):
    sns.heatmap(visualized_all_graph[:,:,i],ax = ax[i],center = 0,square = True,xticklabels=False, yticklabels=False,cbar = False)
    ax[i].set_title(f'Class {i}')
plt.show()