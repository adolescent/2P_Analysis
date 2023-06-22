'''
This script will produce time aligned data for spon analysisi, let's see whether there are difference
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
reducer = ot.Load_Variable(wp,r'Stim_No_ISI_UMAP_Unsup_3d.pkl')
# u = reducer.embedding_
od_frame = ac.Z_Frames[ac.odrun]
orien_frame = ac.Z_Frames[ac.orienrun]
spon_frame = ac.Z_Frames['1-001']
# get label and recombine labels.
all_frame,all_label = ac.Combine_Frame_Labels()
# embedding all frame data into umap space.
u = reducer.transform(all_frame)
#%% show proper embedding of data.
plt.switch_backend('webAgg')
fig,ax = Plot_3D_With_Labels(data = u,labels= all_label)
plt.show()
Save_3D_Gif(ax,fig)
#%% Full fill the space
# get false spiking points.let's set Z range from -1 to 3
all_cells = np.random.randn(100000,678)*10# use even distribution.
false_embeddings = reducer.transform(all_cells)
# plot them into space.
plt.switch_backend('webAgg')
fig,ax = Plot_3D_With_Labels(data = u,labels= all_label)
ax.scatter3D(false_embeddings[:,0],false_embeddings[:,1],false_embeddings[:,2],s = 5,color = 'r')
plt.show()
# Save_3D_Gif(ax,fig)