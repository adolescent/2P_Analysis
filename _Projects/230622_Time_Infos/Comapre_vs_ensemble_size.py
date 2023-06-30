'''

This script will compare the result of SVM with ensemble size only come from cell activation count. What scale of result are found?
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
all_labels = ot.Load_Variable(wp,'spon_svc_labels_0420.pkl')
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
spon_frame = ac.Z_Frames['1-001']
reducer = ot.Load_Variable(wp,'Stim_All_UMAP_Unsup_3d.pkl')
all_stim_frame,all_stim_label = ac.Combine_Frame_Labels(isi = True)
u = reducer.transform(all_stim_frame)
spon_embedding = reducer.transform(spon_frame)
#%% count on cells of each frame.
svc_on_frames = all_labels>0
on_thres = 2
cell_counts = np.array((spon_frame>on_thres).sum(1))
plt.switch_backend('webAgg')
fig,ax = Plot_3D_With_Labels(u,all_stim_label)
plt.show()
#%% plot ensemble num on graph
plt.switch_backend('webAgg')
fig = plt.figure(figsize = (12,10))
fig.tight_layout()
ax = plt.axes(projection='3d')
ax.grid(False)
# ax.set_position([0.1, 0.2, 0.8, 0.7])
ax.scatter3D(spon_embedding[:,0],spon_embedding[:,1],spon_embedding[:,2],s = 3,c = (cell_counts>5)*(cell_counts<10),cmap = 'plasma')
# Save_3D_Gif(ax,fig)
plt.show()
#%% get a set of threshold, get ratio of in-stim ensembles.
thres = np.linspace(4,200,49)
ensemble_counts = np.zeros(thres.shape)
ensemble_repeats = np.zeros(thres.shape)
ensemble_peaklen = np.zeros(thres.shape)
ensemble_repeat_ratio = np.zeros(thres.shape)
for i,c_thres in tqdm(enumerate(thres)):
    event_trains = cell_counts>c_thres
    cutted_events,cutted_len = Label_Event_Cutter(event_trains)
    ensemble_counts[i] = len(cutted_events)
    ensemble_peaklen[i] = cutted_len.mean()
    c_repeat = 0
    for j,c_event in enumerate(cutted_events):
        if svc_on_frames[c_event].sum()>0: # for any event, one frame repeat will be repeat.
            c_repeat += 1
    ensemble_repeats[i] = c_repeat
    ensemble_repeat_ratio[i] = c_repeat/len(cutted_events)
