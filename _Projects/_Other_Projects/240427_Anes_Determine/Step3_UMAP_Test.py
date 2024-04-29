
'''
Simple UMAP and QQ Waketime Estimation.
'''


#%%
import OS_Tools_Kit as ot
import neo
import numpy as np
from neo.io import PlexonIO
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import DBSCAN
import umap
import umap.plot
from Kill_Cache import kill_all_cache

wp = r'D:\#Shu_Data\TailPinch\ChatFlox_Cre\_Stats_All_Points'
all_frames = ot.Load_Variable(wp,'Normalized_All_Raw_Parameters.pkl')
# all_frames = ot.Load_Variable(wp,'All_Raw_Parameters.pkl')
all_case = list(set(all_frames['Case']))
#%% Down sample the data to 10s, so we can save some times.

for i,cloc in enumerate(all_case):
    cloc_data = all_frames.groupby('Case').get_group(all_case[i])
    binnum = len(cloc_data)//10
    c_binned_data = cloc_data.iloc[np.arange(binnum)*10,:].reset_index(drop = True)
    if i == 0:
        down_all_frames = c_binned_data
    else:
        down_all_frames = pd.concat((down_all_frames,c_binned_data))

down_all_frames = down_all_frames.reset_index(drop = True)
ot.Save_Variable(wp,'Down_10_samples',down_all_frames)

#%%
kill_all_cache(r'C:\ProgramData\anaconda3\envs\umapzr')
reducer = umap.UMAP(n_components=2,n_neighbors=500)
# data_frames = down_all_frames.groupby('Case').get_group(all_case[3])
data_frames = down_all_frames
data_frames = data_frames.drop(columns=['Case', 'Time'])
# data_frames = data_frames.drop('Time',axis = 1)
reducer.fit(np.array(data_frames))
ot.Save_Variable(wp,'Raw_UMAP_N500',reducer)
#%% Get UMAP Embedding of all Locations.
scatters = reducer.embedding_
All_Loc_Embeddings = pd.DataFrame(columns = ['Case','UMAP1','UMAP2','Time'])
All_Loc_Embeddings['Case'] = down_all_frames['Case']
All_Loc_Embeddings['Time'] = down_all_frames['Time']
# All_Loc_Embeddings['Case'] = all_frames['Case']
# All_Loc_Embeddings['Time'] = all_frames['Time']
All_Loc_Embeddings['UMAP1'] = scatters[:,0]
All_Loc_Embeddings['UMAP2'] = scatters[:,1]

All_Loc_Embeddings['Time'] = All_Loc_Embeddings['Time'].astype('f8')
All_Loc_Embeddings['UMAP1'] = All_Loc_Embeddings['UMAP1'].astype('f8')
All_Loc_Embeddings['UMAP2'] = All_Loc_Embeddings['UMAP2'].astype('f8')
ot.Save_Variable(wp,'All_Embeddings',All_Loc_Embeddings)

#%% Plot series time train
# cloc_data = down_all_frames.groupby('Case').get_group(all_case[4])
# cloc_data = cloc_data.drop(columns=['Case', 'Time'])
# scatters = reducer.transform(np.array(cloc_data))
# scatters = reducer.embedding_

plt.clf()
plt.cla()
plotable_data = All_Loc_Embeddings[All_Loc_Embeddings['Case'] != '20240129_#243_chat-flox']
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (12,10),dpi = 180,sharex= True)
sns.scatterplot(data = plotable_data,x = 'UMAP1',y = 'UMAP2',hue = 'Time',ax = ax,s = 3,palette = 'hsv',linewidth = 0)
# sns.scatterplot(data = All_Loc_Embeddings,x = 'UMAP1',y = 'UMAP2',hue = 'Case',ax = ax,s = 3,linewidth=0)
# cbar = plt.colorbar(sc)
# cbar.set_label('Time (s)')
ax.set_xlabel('UMAP 1',size = 14)
ax.set_ylabel('UMAP 2',size = 14)
ax.set_title(f'UMAP Embedding of All Data Point',size = 20)
fig.savefig(ot.join(wp,'All_Embedding_Time_without243.svg'))

#%% ######################### Clust and Plot Clusters ##############
# Clust the data to 2 states.
db = DBSCAN(eps=3, min_samples=100).fit(scatters)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
plotable = pd.DataFrame([scatters[:,0],scatters[:,1],labels],index = ['UMAP1','UMAP2','Cluster']).T
# plotable = plotable[plotable['Cluster']!= -1]
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (7,7),dpi = 180,sharex= True)
sns.scatterplot(data = plotable,x ='UMAP1',y = 'UMAP2',hue = 'Cluster',ax = ax,s = 5, palette=sns.color_palette(),linewidth = 0)
ax.legend(markerscale=3)
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('DBSCAN Clusters')
fig.savefig(ot.join(wp,'All_Clusters.svg'))

# clust 0 : awake ;clust 1: anethetized
down_all_frames['Cluster'] = labels

#%% Plot QQ Map of Awake, Anes time, and plot on the graph.
awake_estimation = pd.DataFrame(index = range(len(all_case)),columns = ['Case','Anes','Awake'])

in_thres = 0.002
out_thres = 0.999
for i in range(len(all_case)):
    cloc_data = down_all_frames.groupby('Case').get_group(all_case[i])
    cloc_name = all_case[i].split('\\')[-1]
    c_clusters = np.array(cloc_data['Cluster'])
    c_times = np.array(cloc_data['Time'])

    total_0 = (c_clusters==0).sum()
    total_1 = (c_clusters==1).sum()
    all_0_cdf = np.zeros(len(c_clusters))
    all_1_cdf = np.zeros(len(c_clusters))
    for j in range(len(c_clusters)):
        c_part = c_clusters[:j+1]
        all_0_cdf[j] = (c_part == 0).sum()/total_0
        all_1_cdf[j] = (c_part == 1).sum()/total_1
    
    # plot QQ plot of all graph.
    plt.clf()
    plt.cla()
    fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (5,5),dpi = 180,sharey= True)
    ax.plot(c_times,all_0_cdf,label = 'Awake')
    ax.plot(c_times,all_1_cdf,label = 'Anes')
    ax.set_title(f'{cloc_name} Cultimated Proportion')
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CDF')
    fig.savefig(ot.join(wp,f'{cloc_name}_Frames.svg'))

    # save start and end times.
    start = np.where(all_1_cdf>in_thres)[0][0]*10
    end = (np.where(all_1_cdf>=out_thres)[0][0]+1)*10
    awake_estimation.loc[i,:] = [cloc_name,start,end]
# Add pinch time manually.
awake_estimation['Pinch'] = 0
awake_estimation.loc[0,'Pinch'] = 4*3600+8*60  #736
awake_estimation.loc[1,'Pinch'] = 3*3600+51.5*60  #244
awake_estimation.loc[2,'Pinch'] = 3*3600+42*60-2280  #243
awake_estimation.loc[3,'Pinch'] = 3*3600+37*60  #816
awake_estimation.loc[4,'Pinch'] = 3*3600+54*60   #737
ot.Save_Variable(wp,'Awake_Estimation',awake_estimation)
#%% And Plot all state parameters
for i in range(len(all_case)):
# i = 0
    cloc_data = down_all_frames.groupby('Case').get_group(all_case[i])
    cloc_name = all_case[i].split('\\')[-1]
    columns = ['EEG_Below10Hz','RunSpeed','HR','Pad_Movement','Pluse_Distention','Pupil','Resp','SpO2','BodyTemp']
    c_time = awake_estimation[awake_estimation['Case']==cloc_name]
    plt.clf()
    plt.cla()
    fig,axes = plt.subplots(nrows=9, ncols=1,figsize = (12,12),dpi = 180,sharex= True)
    for i,cpara in enumerate(columns):
        sns.scatterplot(data = cloc_data,x = 'Time',y = cpara,hue = 'Cluster',ax = axes[i],legend = False,s = 3, palette=sns.color_palette(),linewidth = 0)
        axes[i].axvspan(xmin = int(c_time['Anes']),xmax = int(c_time['Awake']),alpha = 0.2,facecolor='y',edgecolor=None)
        axes[i].axvline(x = int(c_time['Pinch']),color = 'red',linestyle='--')
    fig.suptitle(f'{cloc_name} All Biological Parameters',size = 20,y = 1)
    fig.tight_layout()
    # fig.savefig(ot.join(wp,f'{cloc_name}_All_Parameters.png'))
    fig.savefig(ot.join(wp,f'{cloc_name}_All_Parameters.svg'))
# %%## Plot UMAP embeddings as time goes by.
plotable = pd.DataFrame([scatters[:,0],scatters[:,1],labels,down_all_frames['Time'],down_all_frames['Case']],index = ['UMAP1','UMAP2','Cluster','Time','Case']).T

for i in range(len(all_case)):
# i = 0
    cloc_data = plotable.groupby('Case').get_group(all_case[i])
    cloc_name = all_case[i].split('\\')[-1]
    columns = ['UMAP1','UMAP2']
    c_time = awake_estimation[awake_estimation['Case']==cloc_name]
    plt.clf()
    plt.cla()
    fig,axes = plt.subplots(nrows=2, ncols=1,figsize = (8,5),dpi = 180,sharex= True)
    for j,cpara in enumerate(columns):
        sns.scatterplot(data = cloc_data,x = 'Time',y = cpara,hue = 'Cluster',ax = axes[j],legend = False,s = 3, palette=sns.color_palette(),linewidth = 0)
        axes[j].axvspan(xmin = int(c_time['Anes']),xmax = int(c_time['Awake']),alpha = 0.2,facecolor='y',edgecolor=None)
        axes[j].axvline(x = int(c_time['Pinch']),color = 'red',linestyle='--')
    fig.suptitle(f'{cloc_name} UMAP Embedding',size = 14,y = 1)
    fig.tight_layout()
    # fig.savefig(ot.join(wp,f'{cloc_name}_All_Parameters.png'))
    fig.savefig(ot.join(wp,f'{cloc_name}_Embeddings.svg'))



