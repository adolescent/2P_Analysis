'''
This script will do umap on hue data. 
Let's see what will happen.
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
hue_frame = labeled_data_dict['Hue7Orien4'].reset_index(drop = True)
hue_framenum = hue_frame.shape[0]
hue_datasets = np.zeros(shape = (hue_framenum,cell_num),dtype = 'f8')
hue_labels = []
for i in range(hue_framenum):
    c_frame = np.array(hue_frame.loc[i,'Data'])
    hue_datasets[i,:] = c_frame
    hue_labels.append(hue_frame.loc[i,'Color_Label_Num'])
#%% UMAP unsupervised
reducer = umap.UMAP(n_neighbors = 20,min_dist=0.01,n_components=2)
reducer.fit(hue_datasets) # supervised learning on G16 data.
# reducer.fit(g16_datasets) # unsupervised learning on G16 data.
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
umap.plot.points(reducer,ax = ax,labels = np.array(hue_labels),theme = 'fire')
plt.show()
ot.Save_Variable(wp,'Hue_Model_Botheye_unsupervised',reducer)
#%% svc on non-white data. Data After 1079 are white.
hue_no_white = hue_datasets[:1079,:]
hue_labels_no_white = hue_labels[:1079]

#%% SVC 5 fold
hue_embed_unsup = reducer.transform(hue_no_white)
model = svm.SVC()
scores = cross_val_score(model, hue_embed_unsup, hue_labels_no_white, cv=5)
print(f'Score of 5 fold SVC on OD unsupervised : {scores.mean()*100:.2f}%')
#%% embedding spon.
spon_embeddings = reducer.transform(spon_data)
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
umap.plot.points(reducer,ax = ax,labels = np.array(hue_labels),theme = 'fire')
sns.scatterplot(x = spon_embeddings[:,0],y = spon_embeddings[:,1],ax = ax,s = 3,color = 'w')
plt.show()
#%% SVM on unsupervised.
unsup_model = svm.SVC(kernel = 'rbf',probability=True)
predict = unsup_model.fit(hue_embed_unsup,hue_labels_no_white)
spon_probability = unsup_model.predict_proba(spon_embeddings)
#%% visualize
prob_thres = 0.9
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
#%% UMAP on supervised data, exclude white.
reducer2 = umap.UMAP(n_neighbors = 20,min_dist=0.01,n_components=2,target_weight = -1)
reducer2.fit(hue_no_white,hue_labels_no_white) # supervised learning on G16 data.
# reducer2.fit(hue_datasets,hue_labels)
# reducer.fit(g16_datasets) # unsupervised learning on G16 data.
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
umap.plot.points(reducer2,ax = ax,labels = np.array(hue_labels_no_white),theme = 'fire')
# umap.plot.points(reducer2,ax = ax,labels = np.array(hue_labels),theme = 'fire')
plt.show()
ot.Save_Variable(wp,'Hue_Model_Botheye_supervised',reducer2)
#%% embedding spon data 
spon_embeddings = reducer2.transform(spon_data)
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
umap.plot.points(reducer2,ax = ax,labels = np.array(hue_labels_no_white),theme = 'fire')
sns.scatterplot(x = spon_embeddings[:,0],y = spon_embeddings[:,1],ax = ax,s = 3,color = 'w')
plt.show()
#%% SVC on supervised data.
#%% SVM on unsupervised.
sup_model = svm.SVC(kernel = 'rbf',probability=True)
predict = sup_model.fit(reducer2.embedding_,hue_labels_no_white)
spon_probability = sup_model.predict_proba(spon_embeddings)
#%% plot scatters
prob_thres = 0.9
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
#%%
