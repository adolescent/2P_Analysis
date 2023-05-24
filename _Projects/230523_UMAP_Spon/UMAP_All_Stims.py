'''
This script generate all stim umap.

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
# data frame with no isi.
labeled_data_no_isi = labeled_data[labeled_data['Raw_ID']!= '-1']
labeled_data_no_isi = labeled_data_no_isi[labeled_data_no_isi['Color_Label']!= 'White'].reset_index(drop = True)
#%% Get Label ID with np data frame.
frame_num = labeled_data_no_isi.shape[0]
all_stim_frame = np.zeros(shape = (frame_num,cell_num),dtype = 'f8')
all_labels = []
eye_labels = []
orien_labels = []
color_labels = []
for i in range(frame_num):
    c_slide = labeled_data_no_isi.loc[i,:]
    all_stim_frame[i,:] = np.array(c_slide['Data'])
    eye_labels.append(c_slide['OD_Label_Num'])
    orien_labels.append(c_slide['Orien_Label_Num'])
    color_labels.append(c_slide['Color_Label_Num'])
    if c_slide['Color_Label_Num'] != 0:# making color16-19,and 1-4LE,5-8RE,9-16Orien
        c_id = 24+c_slide['Color_Label_Num']
    else:
        c_id = 8*(c_slide['OD_Label_Num']-1)+c_slide['Orien_Label_Num']
    all_labels.append(c_id)
#%% UMAP them.
reducer = umap.UMAP(n_neighbors = 20,min_dist=0.01,n_components=3)
reducer.fit(all_stim_frame) # supervised learning on G16 data.
# reducer.fit(g16_datasets) # unsupervised learning on G16 data.
u = reducer.embedding_
ot.Save_Variable(wp,'Stim_No_ISI_UMAP_Unsup_3d',reducer)
# plt.switch_backend('webAgg')
# fig,ax = plt.subplots(1,figsize = (12,12))
# umap.plot.points(reducer,ax = ax,labels = np.array(all_labels),theme = 'fire')
# plt.show()
#%%
plt.switch_backend('webAgg')
# fig,ax = plt.subplots(1,figsize = (12,12))
# umap.plot.points(reducer,ax = ax,labels = np.array(all_labels),theme = 'inferno')
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.grid(False)
unique_labels = np.unique(eye_labels)
handles = []
# scatter = ax.scatter3D(u[:,0],u[:,1],u[:,2], c=orien_labels,label = orien_labels, cmap='jet', s=3)
all_scatters = []
for label in unique_labels:
    mask = eye_labels == label
    scatter = ax.scatter3D(u[:,0][mask], u[:,1][mask], u[:,2][mask], label=label,s = 5)
    all_scatters.append(scatter)
    handles.append(scatter)
ax.legend(handles=handles)
# ax.set_xlim(2, 8)  # Set X-axis range
# ax.set_ylim(8, 15)  # Set Y-axis range
# ax.set_zlim(3, 11)  # Set Z-axis range
plt.show()
#%% Save current graph as gif
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
x = u[:,0]
y = u[:,1]
z = u[:,2]
n = 2363
def update(frame):
    ax.view_init(elev=10, azim=frame)  # Update the view angle for each frame
    return scatter,

animation = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=50)

# Display the animation
plt.show()
