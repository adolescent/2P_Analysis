'''
This script will answer the question of how network alteration inside spon?
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
from Cell_Class.Plot_Tools import Plot_3D_With_Labels

wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220420_L91\_CAIMAN'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
reducer = ot.Load_Variable(wp,'Stim_No_ISI_UMAP_Unsup_3d.pkl')
u = reducer.embedding_
#%% Get G16/OD data and labels, use them to train an UMAP.
od_frames = ac.Z_Frames[ac.odrun]
orien_frames = ac.Z_Frames[ac.orienrun]
od_labels = ac.OD_Frame_Labels
orien_labels = ac.Orien_Frame_Labels
od_noisi,od_label_noisi = Remove_ISI(od_frames,od_labels)
orien_noisi,orien_label_noisi = Remove_ISI(orien_frames,orien_labels)
spon_frames = ac.Z_Frames['1-001']
#%% Combine two data frame with label.
all_stim_frame = pd.concat([od_noisi,orien_noisi]).reset_index(drop = True)
#generate all real labels. Use 1357 as LE,2468 as RE and 9-16 as Orien,0 as all ISI.
label_od_finale = np.array(od_label_noisi['Raw_ID'])
label_orien_finale = np.zeros(len(orien_label_noisi),dtype = 'i4')
for i in range(len(label_orien_finale)):
    c_label = orien_label_noisi.iloc[i,1]
    if c_label == 0:
        label_orien_finale[i] = 0
    elif c_label == 8 or c_label == 16:
        label_orien_finale[i] = 8+8
    else:
        label_orien_finale[i] = 8+c_label%8
all_label = np.hstack([label_od_finale,label_orien_finale])
#%% we can do umap here.
reducer = umap.UMAP(n_neighbors = 20,min_dist=0.01,n_components=3)
reducer.fit(all_stim_frame)
u = reducer.embedding_
# ot.Save_Variable(wp,'Stim_No_ISI_UMAP_Unsup_3d',reducer) # done.
# Plot 3D
plt.switch_backend('webAgg')
fig,ax = Plot_3D_With_Labels(data = u,labels = all_label)
plt.show()
#%% umap spon data on this space.
spon_embeddings = reducer.transform(spon_frames)
