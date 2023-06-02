'''
This script will only use spon data itself to train a manifold.

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
# data frame with no color.
labeled_data_no_color = labeled_data[labeled_data['From_Stim']!= 'Hue7Orien4']
#%% Generate data with label.
frame_num = labeled_data_no_color.shape[0]
all_stim_frame = np.zeros(shape = (frame_num,cell_num),dtype = 'f8')
all_labels = []
eye_labels = []
orien_labels = []

for i in range(frame_num):
    c_slide = labeled_data_no_color.loc[i,:]
    all_stim_frame[i,:] = np.array(c_slide['Data'])
    eye_labels.append(c_slide['OD_Label_Num'])
    orien_labels.append(c_slide['Orien_Label_Num'])
    # calculate global id.
    if (c_slide['OD_Label_Num'] == 0) or (c_slide['Orien_Label_Num'] == 0):
        whole_id = 0
    else:
        whole_id = 8*(c_slide['OD_Label_Num']-1)+c_slide['Orien_Label_Num']
    all_labels.append(whole_id)
'''
After label generation, we get all eye, orien, both label.
1/3/5/7 as LE, 9/11/13/15 as RE, 17-24 as both eye.
'''    
#%% If load in, run this.
reducer = ot.Load_Variable(wp,'Spon_Only_UMAP_Unsup_3d.pkl')
u = reducer.embedding_
stim_embeddings = reducer.transform(all_stim_frame)
#%% UMAP only spon data to establish manifold.
reducer = umap.UMAP(n_neighbors = 20,min_dist=0.15,n_components=3)
reducer.fit(spon_data) # supervised learning on G16 data.
u = reducer.embedding_
stim_embeddings = reducer.transform(all_stim_frame)
ot.Save_Variable(wp,'Spon_Only_UMAP_Unsup_3d',reducer)
# Plot only spon.
plt.switch_backend('webAgg')
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
ax.grid(False)
ax.scatter3D(u[:,0],u[:,1],u[:,2],s = 3,alpha = 0.5,color = 'b')
plt.show()
#%% Plot spon with stim.
import matplotlib.cm as cm
colors = cm.turbo(np.linspace(0, 1, 10))# colorbars.
plt.clf()
plt.switch_backend('webAgg')
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
ax.grid(False)
# ax.axis('off')
unique_labels = np.unique(orien_labels)
handles = []
all_scatters = []
# scatter0 = ax.scatter3D(u[:,0],u[:,1],u[:,2],s = 1,alpha = 0.3,c = 'b',label=-1)
# handles.append(scatter0)
i = 1
for label in unique_labels:
    mask = orien_labels == label
    scatter = ax.scatter3D(stim_embeddings[:,0][mask], stim_embeddings[:,1][mask], stim_embeddings[:,2][mask], label=label,s = 5,alpha = 1,color = colors[i])
    all_scatters.append(scatter)
    handles.append(scatter)
    i += 1
# ax.scatter3D(shuffle_embeddings[:,0],shuffle_embeddings[:,1],shuffle_embeddings[:,2],s = 3,c = 'black')
# ax.scatter3D(u[:,0],u[:,1],u[:,2],s = 3,alpha = 0.3,c = 'b',label=-1)
ax.legend(handles=handles)
ax.set_ylim(-4, 10)  # Set X-axis range
ax.set_xlim(-0.5, 3)  # Set Y-axis range
ax.set_zlim(2,6)  # Set Z-axis range
ax.view_init(elev=30, azim=0)
plt.show()
#%% Plot Gif
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation

# n = 2363
def update(frame):
    ax.view_init(elev=30, azim=frame)  # Update the view angle for each frame
    return scatter,

animation = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=150)
animation.save('3D_plot.gif', writer='pillow')

#%% Use time sequence as hue to show network variation.
plt.clf()
plt.switch_backend('webAgg')
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
ax.grid(False)
# ax.axis('off')
ax.scatter3D(u[100:,0],u[100:,1],u[100:,2],s = 3,c = list(range(11454)),cmap = 'plasma')
plt.show()