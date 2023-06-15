'''
This will explain the long time trajectory of spon data.
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
from Cell_Class.Plot_Tools import *

# wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220420_L91\_CAIMAN'
wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220914_L85_2P\_CAIMAN'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
reducer = ot.Load_Variable(wp,'Spon_Autosuper.pkl')
u = reducer.embedding_
spon_frame = ac.Z_Frames['1-001']
#%% Get spon data and do umap by it self.
spon_frame = ac.Z_Frames['1-001']
reducer = umap.UMAP(n_neighbors = 30,min_dist=0.01,n_components=3)
reducer.fit(spon_frame)
u = reducer.embedding_
# ot.Save_Variable(wp,'Spon_Autosuper',reducer)
#%%Get weight of each cells ensemble num.
ensemble_scale = (spon_frame>2).sum(1)
ensemble_scale[ensemble_scale>100]=100
# ensemble_scale[ensemble_scale<10]=0

#%% Plot data with time as color.
plt.switch_backend('webAgg')
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
ax.grid(False)
# ax.scatter3D(u[100:,0],u[100:,1],u[100:,2],s = 3,c = list(range(11454)),cmap = 'plasma')
ax.scatter3D(u[100:,0],u[100:,1],u[100:,2],s = 3,c = list(ensemble_scale)[100:],cmap = 'plasma')
plt.show()
#%% Average every 20 points, 
u_cut = u[100:10100]
u_cut_shape = u_cut.reshape((50, 200, 3))
# u_avr = u_cut_shape.mean(1)
u_avr = u_cut_shape[:,0,:]
# u_avr = u_cut[1000:1100]
plt.switch_backend('webAgg')
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
ax.grid(False)
# ax.scatter3D(u_avr[:,0],u_avr[:,1],u_avr[:,2],s = 3,c = list(range(500)),cmap = 'plasma')
ax.scatter3D(u[100:,0],u[100:,1],u[100:,2],s = 3,c = list(range(11454)),cmap = 'plasma',alpha = 0.2)
ax.plot(u_avr[:,0],u_avr[:,1],u_avr[:,2])
plt.show()
# Save_3D_Gif(ax,fig)
#%% Get average of severay segments
L_lag = np.where((u[:,1]<8.7) & (u[:,2]<10.5))[0]
# L_lag = np.where(u[:,0]<3)[0]
# bigger_points = np.where(ensemble_scale>50)[0]
spon_L_lag = spon_frame.iloc[L_lag,:].mean(0)
L_img = Cell_Weight_Visualization(spon_L_lag,ac.all_cell_dic)
plt.switch_backend('webAgg')
sns.heatmap(L_img,center = 0,square = True, xticklabels= False, yticklabels=False,cbar = False)
plt.show()