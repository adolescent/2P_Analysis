'''
This is a V2 point. All process are almost the same.
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

# day_folder = r'D:\ZR\_Data_Temp\Raw_2P_Data\220914_L85_2P'
# ac = Stim_Cells(day_folder=day_folder,od = 6,orien = 7,color = 8)
# ac.Calculate_All()
# ac.Plot_T_Graphs()
# ac.Save_Class()
wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220914_L85_2P\_CAIMAN'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
# reducer = ot.Load_Variable(wp,'Stim_No_ISI_UMAP_Unsup_3d.pkl')
# u = reducer.embedding_
#%% Get spon,color,orien maps.
orien_frame = ac.Z_Frames[ac.orienrun]
color_frame = ac.Z_Frames[ac.colorrun]
spon_frame =ac.Z_Frames['1-001']
orien_labels = ac.Orien_Frame_Labels
color_labels = ac.Color_Frame_Labels
orien_noisi,orien_label_noisi = Remove_ISI(orien_frame,orien_labels)
color_noisi,color_label_noisi = Remove_ISI(color_frame,color_labels)
#%% Combine labels.
all_stim_frame = pd.concat([orien_frame,color_frame]).reset_index(drop = True)
label_orien_finale = np.zeros(len(orien_labels.T),dtype = 'i4')
for i in range(len(label_orien_finale)):
    c_label = orien_labels.T.iloc[i,1]
    if c_label == 0 or c_label == -1:
        label_orien_finale[i] = 0
    elif c_label == 8 or c_label == 16:
        label_orien_finale[i] = 8
    else:
        label_orien_finale[i] = c_label%8
label_color_finale = np.zeros(len(color_labels.T),dtype='i4')
for i in range(len(label_color_finale)):
    c_label = color_labels.T.iloc[i,1]
    if (c_label == 0) or (c_label%7 == 0) or (c_label == -1):
        label_color_finale[i] = 0
    else:
        label_color_finale[i] = c_label%7+8
all_label = np.hstack([label_orien_finale,label_color_finale])
#%%
reducer = umap.UMAP(n_neighbors = 30,min_dist=0.01,n_components=3,target_weight=-1.5)
reducer.fit(all_stim_frame,all_label)
u = reducer.embedding_
# ot.Save_Variable(ac.wp,'Stim_No_ISI_UMAP_Unsup_3d',reducer) # done.
# Plot 3D
plt.switch_backend('webAgg')
fig,ax = Plot_3D_With_Labels(data = u,labels = all_label)
plt.show()
#Save 3D gif.
# Save_3D_Gif(ax,fig)
#%% embed spon on stim space.
spon_embedding = reducer.transform(spon_frame)
plt.switch_backend('webAgg')
fig,ax = Plot_3D_With_Labels(data = u,labels = all_label)
ax.scatter3D(spon_embedding[:,0],spon_embedding[:,1],spon_embedding[:,2],s = 5,c = 'r')
plt.show()
# Save_3D_Gif(ax,fig)
#%% Train an SVC to explain this problem.
# classifier = SVM_Classifier(embeddings = u,label = all_label,C = 1)
classifier = SVM_Classifier(embeddings = u,label = all_label,C = 10)
spon_label_svc = SVC_Fit(classifier,spon_embedding,thres_prob = 0.5)
spon_label_svc[spon_label_svc == 0]=0
plt.switch_backend('webAgg')
fig,ax = Plot_3D_With_Labels(data = spon_embedding,labels = spon_label_svc)
plt.show()
#%% recover maps.
all_avr_spon_clust = Average_Each_Label(spon_frame,spon_label_svc)
plt.switch_backend('webAgg')
fig,ax = Plot_Multi_Subgraphs(graph_frame = all_avr_spon_clust,acd = ac.all_cell_dic,shape = (4,4))
plt.show()