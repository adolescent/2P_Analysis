'''
This script will generate spontaneous embedding on stim-generated space.


'''

#%% Import
from Cell_Class.Stim_Calculators import Stim_Cells
from Cell_Class.Format_Cell import Cell
import OS_Tools_Kit as ot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import cv2
from Kill_Cache import kill_all_cache
from sklearn.model_selection import cross_val_score
from sklearn import svm
import umap
import umap.plot
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import colorsys
import matplotlib as mpl
#%%  ################## Initialization #########################

work_path = r'D:\_Path_For_Figs\S1_PCA_Test'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Datas_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
cc_path = all_path_dic[2]
# Get all stim and spon frames in cell loc.
ac = ot.Load_Variable_v2(cc_path,'Cell_Class.pkl')
spon_frame = ot.Load_Variable(cc_path,'Spon_Before.pkl')
# get all spon graphs with ISI.
all_stim_frame,all_stim_label = ac.Combine_Frame_Labels(color = True)
model = ot.Load_Variable(work_path,'PCA_Model.pkl')
comps_stim = ot.Load_Variable(work_path,'PC_axis.pkl')
coords_stim = ot.Load_Variable(work_path,'All_Stim_Coordinates.pkl')

#%%################# SPON EMBEDDING ########################################
# After stim description,we do spon here. embedding spon onto the stim PC space, and use SVM to classification.
#  train an svc and do reduction.
spon_embeddings = model.transform(spon_frame)
used_dims = coords_stim[:,:20]
classifier,score = SVM_Classifier(used_dims,all_stim_label)
predicted_spon_frame = SVC_Fit(classifier=classifier,data = spon_embeddings[:,1:20],thres_prob=0)
# get all spon embedded frames.
plt.clf()
plt.cla()
u = spon_embeddings
fig = plt.figure(figsize = (10,8))
ax = plt.axes(projection='3d')
ax.grid(False)
sc = ax.scatter3D(u[:,1], u[:,2], u[:,3],s = 5,c = predicted_spon_frame,cmap = 'rainbow')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
animation = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=150)
animation.save(f'Plot_3D.gif', writer='pillow')
