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


day_folder = r'D:\ZR\_Data_Temp\Raw_2P_Data\211015_L84_2P'
ac = Stim_Cells(day_folder,od = False,orien = 2,color = 7,od_type = False)
ac.Calculate_All()
ac.Plot_T_Graphs()
#%% Get several data frames.
G16_Frames = ac.Z_Frames['1-007']
OD_Frames = ac.Z_Frames['1-006']
Color_Frames = ac.Z_Frames['1-008']
Spon_Frames = ac.Z_Frames['1-001']
ac.Save_Class()
#%% get cell datas.
G16_cells = np.array(G16_Frames).T
Spon_Cells = np.array(Spon_Frames).T
orien_labels_raw = ac.all_cell_tunings.loc['Best_Orien',:]
#%% UMAP them.
reducer = umap.UMAP(n_neighbors = 30,n_components = 2)
reducer.fit(Spon_Cells)
u = reducer.embedding_
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (12,12))
umap.plot.points(reducer,ax = ax,theme = 'fire',labels = orien_labels_raw)
plt.show()
