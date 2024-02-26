'''
This is a distance control. Make sure that the corr is not because of 
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
import seaborn as sns
from My_Wheels.Cell_Class.Stim_Calculators import Stim_Cells
from My_Wheels.Cell_Class.Format_Cell import Cell
from Cell_Class.Advanced_Tools import *
from Cell_Class.Plot_Tools import *

wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220420_L91\_CAIMAN'

# wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220914_L85_2P\_CAIMAN'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
reducer = ot.Load_Variable(wp,'Spon_Cell_Seperation_Umap.pkl')
u = reducer.embedding_
spon_frame = ac.Z_Frames['1-001']
#%% show the importance of OD tuning t value.
OD_index = ac.all_cell_tunings.loc['OD',:]
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
ax = plt.scatter(x = u[:,0],y = u[:,1],c = OD_index,cmap = 'bwr')
plt.show()
#%% Calculate distance between 
dist = np.zeros(ac.cellnum)
for i,cc in enumerate(ac.acn):
    c_loc = ac.Cell_Locs[cc]
    c_dist = np.sqrt(c_loc['X']^2+c_loc['Y']^2)
    dist[i] = c_dist
# Try to plot data with c_dist.
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
ax = plt.scatter(x = u[:,0],y = u[:,1],c = dist,cmap = 'bwr')
plt.show()
#%% Get a plot between dist in umap space and dist in real space.
dist_frame = pd.DataFrame(0,columns = ['Cell_Dist','UMAP_Dist'],index = range(int(ac.cellnum*(ac.cellnum-1)/2)))
counter = 0
for i in tqdm(range(ac.cellnum)):
    cell_A_loc = ac.Cell_Locs.iloc[:,i]
    umap_A_loc = u[i,:]
    for j in range(i+1,ac.cellnum):
        cell_B_loc = ac.Cell_Locs.iloc[:,j]
        umap_B_loc = u[j,:]
        c_real_dist = np.linalg.norm(cell_A_loc-cell_B_loc)
        umap_dist = np.linalg.norm(umap_A_loc-umap_B_loc)
        dist_frame.iloc[counter,:] = [c_real_dist,umap_dist]
        counter +=1
#%% plot dist matrix.
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (16,12))
# ax = sns.kdeplot(data = dist_frame,x = 'Cell_Dist',y = 'UMAP_Dist')
ax = sns.scatterplot(data = dist_frame,x = 'Cell_Dist',y = 'UMAP_Dist',s = 3)
plt.show()
#%% Next approuch, let's try to embed umap distance on real space, to see whether the map is similar to OD map.
# data_cen = u.mean(0)
data_cen = np.array([0,0])
# data_cen = u.max(0)
umap_cor = np.zeros(ac.cellnum)
for i in range(ac.cellnum):
    x = u[i,0]-data_cen[0]
    y = u[i,1]-data_cen[1]
    # umap_cor[i] = np.sqrt(x*x+y*y)
    # umap_cor[i] = np.linalg.norm(ac.Cell_Locs[i+1])
    umap_cor[i] = y

# weight_frame = pd.DataFrame(umap_cor,index= ac.acn).T
# try different umap labels.
weight_map = ac.Generate_Weighted_Cell(weight =umap_cor)

weight_map[weight_map == 0] = 9
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (8,8))
# ax = sns.kdeplot(data = dist_frame,x = 'Cell_Dist',y = 'UMAP_Dist')
ax = sns.heatmap(weight_map,center = 9,square = True,xticklabels=False, yticklabels=False)
plt.show()
#%% Another point
from My_Wheels.Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'D:\ZR\_Data_Temp\Raw_2P_Data\220812_L76_2P\220812_L76_stimuli')
# wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220914_L85_2P\_CAIMAN'
day_folder = r'D:\ZR\_Data_Temp\Raw_2P_Data\220812_L76_2P'
ac = Stim_Cells(day_folder = day_folder,od = 6,orien = 2,color = 7)
ac.Calculate_All()
ac.Plot_T_Graphs()
#%% umaps.
Spon_Frames = ac.Z_Frames['1-001']
Spon_Cells = np.array(Spon_Frames).T
reducer = umap.UMAP(n_neighbors = 30,n_components = 2,min_dist= 0.1)
reducer.fit(Spon_Cells)
u = reducer.embedding_
ot.Save_Variable(ac.wp,'Spon_Cell_Seperation_Umap',reducer)
