'''
This script will construct a single UMAP space, embedding Spon and Pinch inside of it.
Only common paras are used.

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
import copy

#%%
pinch_folder = r'E:\临时存储20240819\全参数监测tailpinch\NewData'
spon_folder = r'E:\临时存储20240819\全参数监测自然觉醒\RawData'
spon_subfolders = ot.Get_Subfolders(spon_folder)
pinch_subfolders = ot.Get_Subfolders(pinch_folder)


for i,cloc in tqdm(enumerate(spon_subfolders)):
    cloc_name = cloc.split('\\')[-1]
    cc_frame = ot.Load_Variable_v2(ot.Get_File_Name(cloc,'.pkl')[-1])['All_Frames']
    cc_frame =  (cc_frame-cc_frame.mean(0))/cc_frame.std(0)
    cc_frame['Time']=list(range(len(cc_frame)))
    cc_frame['Case'] = cloc_name
    if i ==0:
        all_spon_frame = copy.deepcopy(cc_frame)
    else:
        all_spon_frame = pd.concat([all_spon_frame,cc_frame])

for i,cloc in tqdm(enumerate(pinch_subfolders)):
    cloc_name = cloc.split('\\')[-1]
    cc_frame = ot.Load_Variable_v2(ot.Get_File_Name(cloc,'.pkl')[-1])['All_Frames']
    cc_frame =  (cc_frame-cc_frame.mean(0))/cc_frame.std(0)
    cc_frame['Time']=list(range(len(cc_frame)))
    cc_frame['Case'] = cloc_name
    if i ==0:
        all_pinch_frame = copy.deepcopy(cc_frame)
    else:
        all_pinch_frame = pd.concat([all_pinch_frame,cc_frame])

all_pinch_frame['Type'] = 'Pinch'
all_spon_frame['Type'] = 'Spon'
all_frames = pd.concat([all_pinch_frame,all_spon_frame])
all_frames = all_frames.dropna(how='any',axis = 1)
all_case = list(set(all_frames['Case']))

#%% 
# wp = r'D:\#Shu_Data\UMAP_Pinch_Spon_Compare'


# pinch_frames = ot.Load_Variable(r'D:\#Shu_Data\TailPinch\ChatFlox_Cre\_Stats_All_Points','Normalized_All_Raw_Parameters.pkl')
# spon_frames = ot.Load_Variable(r'D:\#Shu_Data\Data_SpontaneousWakeUp\_Stats_All_Points','Normalized_All_Raw_Parameters.pkl')
# # spon_frames = ot.Load_Variable(r'E:\临时存储20240819\全参数监测自然觉醒\RawData\20240708_#1085\All_Params_1Hz_#1085.pkl')
# # spon_frames = spon_frames['All_Frames']
# # spon_frames = (spon_frames-spon_frames.mean(0))/spon_frames.std(0)

# # all_frames = ot.Load_Variable(wp,'All_Raw_Parameters.pkl')
# # all_case = list(set(all_frames['Case']))

# # concat frames, keep only common paras
# pinch_frames['Type'] = 'Pinch'
# spon_frames['Type'] = 'Spon'
# all_frames = pd.concat([pinch_frames,spon_frames])
# all_frames = all_frames.dropna(how='any',axis = 1)
# all_case = list(set(all_frames['Case']))
#%% down sample data, umap is a slow calculation, we can use 10s as a bin.
bin_width = 10

for i,cloc in enumerate(all_case):
    cloc_data = all_frames.groupby('Case').get_group(all_case[i])
    binnum = len(cloc_data)//bin_width
    c_binned_data = cloc_data.iloc[np.arange(binnum)*bin_width,:].reset_index(drop = True)
    if i == 0:
        down_all_frames = c_binned_data
    else:
        down_all_frames = pd.concat((down_all_frames,c_binned_data))
down_all_frames = down_all_frames.reset_index(drop = True)

#%% UMAP Data. Paras need to be adjusted.
n_nei = 1000
n_comp = 2

kill_all_cache(r'C:\ProgramData\anaconda3\envs\umapzr')
reducer = umap.UMAP(n_components=n_comp,n_neighbors=n_nei)
# reducer = umap.UMAP(n_components=n_comp,n_neighbors=n_nei,min_dist=0.2)
# data_frames = down_all_frames.groupby('Case').get_group(all_case[3])
umapable_frames = copy.deepcopy(down_all_frames)
# umapable_frames = umapable_frames[umapable_frames['Type']=='Spon']
umapable_frames = umapable_frames.drop(columns=['Case', 'Time','Type'])
# data_frames = data_frames.drop('Time',axis = 1)
reducer.fit(np.array(umapable_frames))
# ot.Save_Variable(wp,'UMAP_2D_1000N',reducer)
# scatters = reducer.transform(np.array(down_all_frames.drop(columns = ['Case','Time','Type'])))
#%% Plot all time data point of a single pooint.
scatters = reducer.transform(all_frames[all_frames['Case']==all_case[0]].drop(columns = ['Case','Time','Type']))
#%% Plot single location's all response.
All_Loc_Embeddings = pd.DataFrame(columns = ['UMAP1','UMAP2','Time'])
All_Loc_Embeddings['UMAP1'] = scatters[:,0]
All_Loc_Embeddings['UMAP2'] = scatters[:,1]
All_Loc_Embeddings['Time'] = list(range(len(scatters)))

plt.clf()
plt.cla()
plotable_data = All_Loc_Embeddings
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (12,10),dpi = 180,sharex= True)
sns.scatterplot(data = plotable_data,x = 'UMAP1',y = 'UMAP2',hue = 'Time',ax = ax,s = 3,palette = 'hsv',linewidth = 0)
ax.set_xlabel('UMAP 1',size = 14)
ax.set_ylabel('UMAP 2',size = 14)
# ax.set_title(f'UMAP Embedding of Spontaneous Wakeup',size = 20)
fig.savefig(ot.join(wp,'All_Embedding_Pinch.svg'))
#%%
'''
Till here, calculation is done. We will Visualize the data.

'''
# scatters = reducer.embedding_
scatters = reducer.transform(down_all_frames.drop(columns = ['Case','Time','Type']))
All_Loc_Embeddings = pd.DataFrame(columns = ['Case','UMAP1','UMAP2','Time','Type'])
All_Loc_Embeddings['Case'] = down_all_frames['Case']
All_Loc_Embeddings['Time'] = down_all_frames['Time']
All_Loc_Embeddings['Type'] = down_all_frames['Type']
# All_Loc_Embeddings['Case'] = all_frames['Case']
# All_Loc_Embeddings['Time'] = all_frames['Time']
All_Loc_Embeddings['UMAP1'] = scatters[:,0]
All_Loc_Embeddings['UMAP2'] = scatters[:,1]



All_Loc_Embeddings['Time'] = All_Loc_Embeddings['Time'].astype('f8')
All_Loc_Embeddings['UMAP1'] = All_Loc_Embeddings['UMAP1'].astype('f8')
All_Loc_Embeddings['UMAP2'] = All_Loc_Embeddings['UMAP2'].astype('f8')

# Plot parts
plt.clf()
plt.cla()
plotable_data = All_Loc_Embeddings[All_Loc_Embeddings['Case'] != '20240129_#243_chat-flox']
# plotable_data = plotable_data[plotable_data['Type']=='Spon']
plotable_data = plotable_data[plotable_data['Type']=='Pinch']
# plotable_data = plotable_data[plotable_data['Case']=='20240708_#1085']


fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (12,10),dpi = 180,sharex= True)
sns.scatterplot(data = plotable_data,x = 'UMAP1',y = 'UMAP2',hue = 'Time',ax = ax,s = 3,palette = 'hsv',linewidth = 0)
# sns.scatterplot(data = All_Loc_Embeddings,x = 'UMAP1',y = 'UMAP2',hue = 'Case',ax = ax,s = 3,linewidth=0)
# cbar = plt.colorbar(sc)
# cbar.set_label('Time (s)')
ax.set_xlabel('UMAP 1',size = 14)
ax.set_ylabel('UMAP 2',size = 14)
ax.set_title(f'UMAP Embedding of Spontaneous Wakeup',size = 20)
# ax.set_xlim(-5,15)
# ax.set_ylim(-3,15)


fig.savefig(ot.join(wp,'All_Embedding_Spon.svg'))

#%% Let's try some interesting 3D things!
'''
Here we try to plot UMAP in 3D.
'''
n_nei = 100
n_comp = 3
kill_all_cache(r'C:\ProgramData\anaconda3\envs\umapzr')
reducer = umap.UMAP(n_components=n_comp,n_neighbors=n_nei)
# data_frames = down_all_frames.groupby('Case').get_group(all_case[3])
umapable_frames = copy.deepcopy(down_all_frames)
umapable_frames = umapable_frames.drop(columns=['Case', 'Time','Type'])
# data_frames = data_frames.drop('Time',axis = 1)
reducer.fit(np.array(umapable_frames))
ot.Save_Variable(wp,'UMAP_3D_100N',reducer)
#%% And plot 3D here.
scatters = reducer.embedding_
All_Loc_Embeddings = pd.DataFrame(columns = ['Case','UMAP1','UMAP2','UMAP3','Time','Type'])
All_Loc_Embeddings['Case'] = down_all_frames['Case']
All_Loc_Embeddings['Time'] = down_all_frames['Time']
All_Loc_Embeddings['Type'] = down_all_frames['Type']
# All_Loc_Embeddings['Case'] = all_frames['Case']
# All_Loc_Embeddings['Time'] = all_frames['Time']
All_Loc_Embeddings['UMAP1'] = scatters[:,0]
All_Loc_Embeddings['UMAP2'] = scatters[:,1]
All_Loc_Embeddings['UMAP3'] = scatters[:,2]

All_Loc_Embeddings['Time'] = All_Loc_Embeddings['Time'].astype('f8')
All_Loc_Embeddings['UMAP1'] = All_Loc_Embeddings['UMAP1'].astype('f8')
All_Loc_Embeddings['UMAP2'] = All_Loc_Embeddings['UMAP2'].astype('f8')
All_Loc_Embeddings['UMAP3'] = All_Loc_Embeddings['UMAP3'].astype('f8')

#%% for 3D plot, we need to normalize time here..
max_time_per_case = All_Loc_Embeddings.groupby('Case')['Time'].transform('max')
# Normalize the Time column using the maximum time for each case
All_Loc_Embeddings['NormalizedTime'] =All_Loc_Embeddings['Time'] / max_time_per_case

for i,c_loc in enumerate(all_case):
    cloc_data = All_Loc_Embeddings[All_Loc_Embeddings['Case'] == c_loc]
    print(cloc_data['NormalizedTime'].max())


#%%
import colorsys

# get all colors
plotable_data = All_Loc_Embeddings[All_Loc_Embeddings['Case'] != '20240129_#243_chat-flox']
# plotable_data = plotable_data[plotable_data['Type']=='Spon']
plotable_data = plotable_data[plotable_data['Type']=='Pinch']
u = np.array(plotable_data[['UMAP1','UMAP2','UMAP3']])

all_color_info = np.zeros(shape = (len(plotable_data),3))
for i in range(len(plotable_data)):
    c_time = plotable_data.iloc[i,-1]
    all_color_info[i:] = colorsys.hls_to_rgb(c_time,0.5,1)


plt.clf()
plt.cla()

fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (8,4),dpi = 180,subplot_kw=dict(projection='3d'))
orien_elev = 25
orien_azim = 170
ax.scatter3D(u[:,0],u[:,1],u[:,2],s = 0.3,c = all_color_info)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
fig.tight_layout()

fig.savefig(ot.join(wp,'All_Embedding_Pitch_3D.svg'))
