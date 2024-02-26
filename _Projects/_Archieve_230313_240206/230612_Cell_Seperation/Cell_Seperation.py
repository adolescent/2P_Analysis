'''
This script will seperate different cells in umap space.

'''
#%%
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
import random
from My_Wheels.Cell_Class.Stim_Calculators import Stim_Cells
from My_Wheels.Cell_Class.Format_Cell import Cell
from Cell_Class.Advanced_Tools import *


day_folder = r'D:\ZR\_Data_Temp\Raw_2P_Data\220420_L91'
ac = Stim_Cells(day_folder,od = 6,orien = 7,color = 8)
ac.Calculate_All()
ac.Plot_T_Graphs()
#%% If load in, run this.
wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220420_L91\_CAIMAN'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
#%% Get several data frames.
G16_Frames = ac.Z_Frames['1-007']
OD_Frames = ac.Z_Frames['1-006']
Color_Frames = ac.Z_Frames['1-008']
Spon_Frames = ac.Z_Frames['1-001']
# ac.Save_Class()
#%% get cell datas.
G16_cells = np.array(G16_Frames).T
Spon_Cells = np.array(Spon_Frames).T
orien_labels_raw = ac.all_cell_tunings.loc['Best_Orien',:]
od_labels_raw = ac.all_cell_tunings.loc['Best_Eye',:]
#%% Get Combined_Cell tunings.
orien_labels = np.zeros(ac.cellnum,dtype='i4')
od_labels = np.zeros(ac.cellnum,dtype = 'i4')
for i,cc in enumerate(ac.acn):
    if od_labels_raw[cc] == 'LE':
        od_labels[i] = 1
    elif od_labels_raw[cc] == 'RE':
        od_labels[i] = 2
    else:
        od_labels[i] = 3
    if orien_labels_raw[cc] == 'False':
        orien_labels[i] = 9
    else:
        orien_labels[i] = int(1+float(orien_labels_raw[cc][5:])/22.5)
# get combined_label
combined_labels = (od_labels-1)*9+orien_labels

#%% UMAP them.
reducer = umap.UMAP(n_neighbors = 30,n_components = 2,min_dist= 0.1)
reducer.fit(Spon_Cells)
u = reducer.embedding_
ot.Save_Variable(ac.wp,'Spon_Cell_Seperation_Umap',reducer)
#%%
reducer = ot.Load_Variable(ac.wp,'Spon_Cell_Seperation_Umap.pkl')
u = reducer.embedding_
#%% Plots
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
umap.plot.points(reducer,ax = ax,theme = 'fire',labels = od_labels)
plt.show()
#%% OD index map. Index is bad so we use T value.
OD_index = ac.all_cell_tunings.loc['OD',:]
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
ax = plt.scatter(x = u[:,0],y = u[:,1],c = OD_index,cmap = 'coolwarm')
plt.show()
#%% Plot best oriens with data.
ac.all_cell_tunings.loc['x'] = u[:,0]
ac.all_cell_tunings.loc['y'] = u[:,1]
# get all tunings.
non_orien_list = np.where(ac.all_cell_tunings.T['Best_Orien']=='False')
all_orien_frames = ac.all_cell_tunings.drop(ac.all_cell_tunings.T.index[non_orien_list],axis = 1) # keep origin frame and drop a new one.
#%% get sin value of orientations.
for i,cc in enumerate(all_orien_frames.columns):
    c_orien = float(all_orien_frames.loc['Best_Orien',cc][5:])
    c_sin = np.sin(c_orien*np.pi/180)
    all_orien_frames.loc['Orien_sin',cc] = c_sin
    all_orien_frames.loc['Orien_abs',cc] = min(180-c_orien,c_orien)
    all_orien_frames.loc['Best_Orien_num',cc] = c_orien
    all_orien_frames.loc['Best_Orien_t',cc] = all_orien_frames.loc[all_orien_frames.loc['Best_Orien',cc]+'-0',cc]
all_orien_frames.loc['Orien_index_new'] = np.clip(all_orien_frames.loc['Orien_index'],-1,1 )
#%% plot oriens.
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
sns.scatterplot(data = all_orien_frames.T,x = 'x',y = 'y',hue = 'Best_Orien_num',palette = 'twilight_shifted',size = 'Best_Orien_t',sizes = (15,100),legend = 'brief')
plt.show()
#%% Let's try if PCA can get similar results.
_,pc_embeds,_ = Z_PCA(ac.Z_Frames['1-001'],sample = 'Cell')
all_pc_frame = pd.DataFrame(columns=ac.acn)
for i,cc in enumerate(ac.acn):
    c_pc = pc_embeds[i,:] # all pc of current cell
    for j in range(len(c_pc)):
        all_pc_frame.loc['PC'+str(1000+j+1)[1:],cc] = c_pc[j]
#%% concat this with act
frame_with_pc = pd.concat([all_orien_frames, all_pc_frame], axis=0)
#%%
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
sns.scatterplot(data = frame_with_pc.T,x = 'PC001',y = 'PC002',hue = 'Best_Orien_num',palette = 'twilight_shifted')
plt.show()