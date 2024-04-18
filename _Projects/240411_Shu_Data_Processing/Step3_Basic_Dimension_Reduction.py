'''
Read in all same captured data.

'''


#%%
import numpy as np
from scipy.interpolate import interp1d
import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pywt
from tqdm import tqdm
from scipy import signal
from Advanced_Tools import Z_PCA

wp = r'D:\_Shu_Data\#244_Py_Data'
all_data = ot.Load_Variable(wp,'All_Samples_1Hz.pkl')
all_data_frame = all_data['All_Frames']
data_frame_normed = (all_data_frame-all_data_frame.mean(0))/all_data_frame.std(0)
#%% Plot All Raw Datas.
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=8, ncols=1,figsize = (12,15),dpi = 144,sharex= True)
for i in range(8):
    ax[i].plot(all_data_frame.iloc[:,i],color=plt.cm.tab10(i))
    ax[i].set_ylabel(all_data_frame.columns[i])
ax[7].set_xticks(np.linspace(0,330,12)*60)
ax[7].set_xticklabels(np.linspace(0,330,12))
fig.suptitle('All Parameters',y = 1.01,size = 20)
fig.tight_layout()

#%% Do PCA on data.
# pc_comps,coords,model = Z_PCA(all_data_frame.iloc[:,:7],'Frame',6)
pc_comps,coords,model = Z_PCA(data_frame_normed.iloc[:,:7],'Frame',6)
model_var_ratio = model.explained_variance_ratio_
plt.plot(coords[:,0])
# plt.scatter(coords[:,0],coords[:,1],s = 1,c = range(len(coords[:,0])),cmap = 'jet')
#%% Plot PC explained variance here.
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (6,4),dpi = 144)
sns.barplot(y = model_var_ratio*100,x = np.arange(1,len(model_var_ratio)+1),ax = ax)
ax.set_xlabel('PC',size = 12)
ax.set_ylabel('Explained Variance (%)',size = 12)
ax.set_title('Each PC explained Variance',size = 14)
#%% Plot PC1 and PC2 plots.
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=2, ncols=1,figsize = (12,5),dpi = 144,sharex= True)
ax[0].plot(coords[:,0],color=plt.cm.tab10(1))
ax[1].plot(coords[:,1],color=plt.cm.tab10(2))

ax[0].set_title('PC Power With Time')
ax[0].set_ylabel('PC1 Power')
ax[1].set_ylabel('PC2 Power')
ax[1].set_xticks(np.linspace(0,330,12)*60)
ax[1].set_xticklabels(np.linspace(0,330,12))
ax[1].set_xlabel('Time (min)')

#%% Plot Element Contribution To PC.
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (7,4),dpi = 144)
sns.barplot(y = pc_comps[2,:],x = np.arange(len(pc_comps[0,:])),ax = ax,width=0.5)
ax.set_xlabel('Parameter',size = 12)
ax.set_ylabel('Weight',size = 12)
ax.set_xticklabels(list(data_frame_normed.iloc[:,:7].columns),size = 8)
ax.set_title('Each Parameter Contribution',size = 14)
# ax.set_ylim(0,0.6)
#%% Scatter data of thres PC value >-0.5

# for i,c_color in enumerate(thresed_id):
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (7,5),dpi = 144,sharex= True)
# ax.scatter(coords[:,0],coords[:,1],s = 1,c = range(len(coords)),cmap = 'bwr',alpha = 0.5)
frame =pd.DataFrame([coords[:,0],coords[:,1],coords[:,2]]).T
frame = frame.sample(1000)
# sns.kdeplot(data=frame,
#     x=0, y=1,
#     fill=False, thresh=0.1, levels=5, bw_adjust=1,
#     cmap='hot',ax = ax
# )
sns.histplot(data = frame,x = 0,y = 1,ax = ax,bins = list(np.linspace(-3,6,50)),cmap = 'inferno')
ax.set_title('PCA Coordinate Density')
ax.set_xlabel('PC1 Coord')
ax.set_ylabel('PC2 Coord')
ax.set_xlim(-3,6)
ax.set_ylim(-2.5,3)


#%% Plot All PC weights
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=6, ncols=1,figsize = (15,12),dpi = 144,sharex= True)
for i in range(6):
    ax[i].plot(coords[:,i],color=plt.cm.tab10(i))
    ax[i].set_ylabel(f'PC {i+1}')
ax[5].set_xticks(np.linspace(0,330,12)*60)
ax[5].set_xticklabels(np.linspace(0,330,12))
fig.suptitle('All PC Weights',y = 1.01,size = 20)
fig.tight_layout()

#%% Let's try some UMAP.
import umap
import umap.plot
#%%
reducer = umap.UMAP(n_components=2,n_neighbors=1000)
reducer.fit(data_frame_normed.iloc[:,:7])
# reducer.fit(all_data_frame.iloc[:,:7])
#%%
scatters = reducer.embedding_
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=1.5, min_samples=100).fit(scatters)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
#%% Plot series time train
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (10,7),dpi = 144,sharex= True)
sc = plt.scatter(scatters[:,0],scatters[:,1],s = 1,c = range(len(scatters)),cmap='hsv')
cbar = plt.colorbar(sc)
cbar.set_label('Time (s)')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('UMAP Embedding of #244 Physical States')

#%% plot dbscan cluster.
# plt.plot(scatters[:,0],scatters[:,1])
plotable = pd.DataFrame([scatters[:,0],scatters[:,1],labels],index = ['UMAP1','UMAP2','Cluster']).T
plotable = plotable[plotable['Cluster']!= -1]
#%%
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (7,7),dpi = 180,sharex= True)
sns.scatterplot(data = plotable,x ='UMAP1',y = 'UMAP2',hue = 'Cluster',ax = ax,s = 5, palette=sns.color_palette())
ax.legend(markerscale=3)
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('DBSCAN Clusters')

#%% Plot Cluster Times
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (12,5),dpi = 180,sharex= True)

plotable_data = pd.DataFrame([scatters[:,0],scatters[:,1],labels]).T
plotable_data[2] = plotable_data[2].replace(-1,999)
sns.scatterplot(plotable_data,x = range(len(plotable_data)),y = coords[:,0],hue = 2,palette = sns.color_palette(),s = 2,ax = ax)
ax.legend(markerscale=3)

ax.set_xticks(np.linspace(0,330,12)*60)
ax.set_xticklabels(np.linspace(0,330,12))
ax.set_xlabel('Time (min)')
ax.set_ylabel('PC1 Weights')
ax.set_title('Cluster on Times')