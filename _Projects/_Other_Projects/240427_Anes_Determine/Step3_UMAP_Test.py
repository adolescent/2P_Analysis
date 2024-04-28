
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
all_frames = ot.Load_Variable(wp,'All_Raw_Parameters.pkl')
all_case = list(set(all_frames['Case']))

#%%
kill_all_cache(r'C:\ProgramData\anaconda3\envs\umapzr')
reducer = umap.UMAP(n_components=2,n_neighbors=100)
reducer.fit(np.array(all_frames.iloc[:,:9]))
ot.Save_Variable(wp,'Raw_UMAP_N100',reducer)
#%% Plot series time train
# cloc_data = all_frames.groupby('Case').getgroup(all_case[0])
scatters = reducer.embedding_
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (10,7),dpi = 144,sharex= True)
sc = plt.scatter(scatters[:,0],scatters[:,1],s = 1)
cbar = plt.colorbar(sc)
cbar.set_label('Time (s)')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title(f'UMAP Embedding of All States')
