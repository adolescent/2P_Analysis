'''
Repeat all procedure, but using only supervised umap as labeler.
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
from scipy.stats import pearsonr
import scipy.stats as stats

wp = r'D:\ZR\_Data_Temp\_All_Cell_Classes\220420_L91'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
spon_frame = ac.Z_Frames['1-001']
kill_all_cache(r'C:\ProgramData\anaconda3\envs\umapzr')
all_stim_frame,all_stim_label = ac.Combine_Frame_Labels(isi = True)
#%% train an supervised svm.
reducer_sup = umap.UMAP(n_components= 3,n_neighbors=20,target_weight=0)
reducer_sup.fit(all_stim_frame,all_stim_label)
u = reducer_sup.embedding_
spon_embedding = reducer_sup.transform(spon_frame)
#plot 3d
plt.switch_backend('webAgg')
fig,ax =Plot_3D_With_Labels(u,all_stim_label)
# ax.scatter3D(spon_embedding[:,0],spon_embedding[:,1],spon_embedding[:,2],c = 'r',s = 3)
plt.show()
# Save_3D_Gif(ax,fig)
#%% and train svm
classifier,score = SVM_Classifier(u,all_stim_label)
spon_svm_label = SVC_Fit(classifier,spon_embedding,0)
plt.switch_backend('webAgg')
fig,ax =Plot_3D_With_Labels(spon_embedding,spon_svm_label)
plt.show()
event,length = Label_Event_Cutter(spon_svm_label>0)
#%% Shuffle to random combine OD/Orien networks.
spike_series_od = (spon_svm_label>0)*(spon_svm_label<9)
spike_series_orien = (spon_svm_label>0)*(spon_svm_label>8)
od_events,od_event_len = Label_Event_Cutter(spike_series_od)
orien_events,orien_event_len = Label_Event_Cutter(spike_series_orien)
all_events,all_len = Label_Event_Cutter(spon_svm_label>0)
# shuffle.
N = 100000
shuffle_shape = pd.DataFrame(0,index = range(N),columns=['Coactivation','mean_length','std'])
for i in tqdm(range(N)): # cycle shuffle
    # generate random OD series
    