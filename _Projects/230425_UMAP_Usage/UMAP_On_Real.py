'''
Try use UMAP on our data, test each usage.
'''

#%%
import OS_Tools_Kit as ot
import numpy as np
import umap
import umap.plot
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from Kill_Cache import kill_all_cache
from tqdm import tqdm
from sklearn.decomposition import PCA

conda_path = r'C:\ProgramData\anaconda3\envs\umapzr' # if kernel dies, kill them all.
kill_all_cache(conda_path)
wp = r'D:\ZR\_Data_Temp\_Temp_Data\220711_temp'
umap_folder = ot.join(wp,'UMAP_Datas')
ot.mkdir(umap_folder)
cd91 = ot.Load_Variable(wp,'All_Series_Dic91.pkl')
#%% get spon data frame, F value,dF value, dff values(least 10%),dff values(mean),z value.
cell_names = list(cd91.keys())
frame_nums = len(cd91[1]['1-001'])
from Filters import Signal_Filter
Frame_Raw_Run01 = pd.DataFrame(columns = cell_names,index=list(range(frame_nums)))
Frame_dff_mean_Run01 = pd.DataFrame(columns = cell_names,index=list(range(frame_nums)))
Frame_dff_least_Run01 = pd.DataFrame(columns = cell_names,index=list(range(frame_nums)))
Frame_Z_Run01 = pd.DataFrame(columns = cell_names,index=list(range(frame_nums)))
Frame_Z_least_Run01 = pd.DataFrame(columns = cell_names,index=list(range(frame_nums)))

filt_para = (0.005,0.3)
fps = 1.301
for i,cc in tqdm(enumerate(cell_names)):
    cc_series = cd91[cc]['1-001']
    filted_series = Signal_Filter(cc_series,order=5,filter_para = (filt_para[0]*2/fps,filt_para[1]*2/fps))
    base_mean = filted_series.mean()
    # get least base
    base_least_num = int(frame_nums*0.1)
    base_id = np.argpartition(filted_series, base_least_num)[:base_least_num]
    base_least = filted_series[base_id].mean()
    # get dff series and least dff series.
    dff_series = (filted_series-base_mean)/base_mean
    dff_series_least = (filted_series-base_least)/base_least
    z_series = dff_series/dff_series.std()
    z_series_least = dff_series_least/dff_series_least.std()
    # write each columns into data frame.
    Frame_Raw_Run01.loc[:,cc] = filted_series
    Frame_dff_mean_Run01.loc[:,cc] = dff_series
    Frame_dff_least_Run01.loc[:,cc] = dff_series_least
    Frame_Z_Run01.loc[:,cc] = z_series
    Frame_Z_least_Run01.loc[:,cc] = z_series_least
#%% save variables.
ot.Save_Variable(umap_folder,'Raw_F_Run01',Frame_Raw_Run01)
ot.Save_Variable(umap_folder,'Dff_mean_Run01',Frame_dff_mean_Run01)
ot.Save_Variable(umap_folder,'Dff_least_Run01',Frame_dff_least_Run01)
ot.Save_Variable(umap_folder,'Z_mean_Run01',Frame_Z_Run01)
ot.Save_Variable(umap_folder,'Z_least_Run01',Frame_Z_least_Run01)
#%% If saved, load data.
Frame_dff_mean_Run01 = ot.Load_Variable(umap_folder,'Dff_mean_Run01.pkl')
Frame_Z_mean_Run01 = ot.Load_Variable(umap_folder,'Z_mean_Run01.pkl')
Frame_dff_least_Run01 = ot.Load_Variable(umap_folder,'Dff_least_Run01.pkl')

#%% Plotter
plt.switch_backend('webAgg')
plt.plot(Frame_dff_mean_Run01.loc[:,510])
plt.show()

#%% Do UMAP for data.
# save and load use simple pikle(ot.save works)
# Data type shall be in shape N_Sample*M_Dim

reducer = umap.UMAP(n_neighbors = 10,min_dist=0.01,n_components=3)
# try original dF/F data by tradition.
# dff_data = np.array(Frame_dff_mean_Run01)[2000:10000,:]
dff_data = np.array(Frame_Z_mean_Run01)
# dff_data = PCA(n_components=200).fit_transform(dff_data)
# dff_data = mm1.T
# reducer.fit(dff_data)
reducer.fit(a[:,2:50])
#%% plot data
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
umap.plot.points(reducer,ax = ax)
plt.show()
#%% Plot 3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
u = reducer.embedding_
plt.switch_backend('webAgg')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(u[:,0],u[:,1],u[:,2])
plt.show()


#%% plot connectivity
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (10,10))
umap.plot.connectivity(reducer,show_points= True)
# umap.plot.connectivity(reducer,edge_bundling='hammer')
plt.show()
#%% Diagnostic
plt.switch_backend('webAgg')
umap.plot.diagnostic(reducer, diagnostic_type='pca')
plt.show()
# #%% Try on rivalry data.
# import h5py
# arrays = {}
# f = h5py.File(r'D:\ZR\_Data_Temp\BRdata.mat')
# for k, v in f.items():
#     arrays[k] = np.array(v)
# mm1 = arrays['TestData']

#%% Cluster test.
import sklearn.cluster as cluster
kmeans_labels = cluster.KMeans(n_clusters=9).fit_predict(dff_data)
standard_embedding = reducer.embedding_

plt.switch_backend('webAgg')
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=kmeans_labels, s=3, cmap='Spectral')
plt.show()
