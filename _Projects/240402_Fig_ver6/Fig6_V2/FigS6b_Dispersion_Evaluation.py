'''
Use Euler dist to evalutate stim and spon data dispersion.

'''


#%%
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
from Cell_Class.UMAP_Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *



all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)

import warnings
warnings.filterwarnings("ignore")




#%% P1 calculate all dist in all dimemsions.
all_dist = pd.DataFrame(columns = ['Loc','Dim','DataType','Var'])
for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = np.array(ot.Load_Variable(cloc,'Spon_Before.pkl'))
    spon_s_phase = Spon_Shuffler(c_spon,method='phase')
    # pcnum = PCNum_Determine(c_spon,sample='Frame',thres = 0.5)
    
    pcnum = 10
    spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=c_spon,sample='Frame',pcnum=pcnum)
    analyzer = UMAP_Analyzer(ac = ac,umap_model=spon_models,spon_frame=c_spon,od = 0, color = 0,orien = 1)
    analyzer.Train_SVM_Classifier()
    g16_embed = analyzer.stim_embeddings
    spon_embed_s = spon_models.transform(spon_s_phase)

    spon_range_per_dim = np.std(spon_coords, axis=0)
    spon_range_per_dim_s = np.std(spon_embed_s, axis=0)
    stim_range_per_dim = np.std(g16_embed, axis=0)

    # save data into frame.
    for j in range(10):
        all_dist.loc[len(all_dist),:] = [cloc_name,j+1,'Spon',spon_range_per_dim[j]]
        all_dist.loc[len(all_dist),:] = [cloc_name,j+1,'Stim',stim_range_per_dim[j]]
        all_dist.loc[len(all_dist),:] = [cloc_name,j+1,'Shuffle',spon_range_per_dim_s[j]]


#%% Ver2 calculate dFF std.
all_dist = pd.DataFrame(columns = ['Loc','Dim','DataType','Var'])
for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    c_orien_dff = ac.Get_dFF_Frames(ac.orienrun)
    c_spon_dff = ac.Get_dFF_Frames(runname='1-001',start = c_spon.index[0],stop = c_spon.index[-1])
    spon_s_phase = Spon_Shuffler(c_spon_dff,method='phase')
    # pcnum = PCNum_Determine(c_spon,sample='Frame',thres = 0.5)
    
    pcnum = 10
    spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=c_spon_dff,sample='Frame',pcnum=pcnum)
    g16_embed = spon_models.transform(c_orien_dff)
    spon_embed_s = spon_models.transform(spon_s_phase)

    spon_range_per_dim = np.std(spon_coords, axis=0)
    spon_range_per_dim_s = np.std(spon_embed_s, axis=0)
    stim_range_per_dim = np.std(g16_embed, axis=0)

    # save data into frame.
    for j in range(10):
        all_dist.loc[len(all_dist),:] = [cloc_name,j+1,'Spon',spon_range_per_dim[j]]
        all_dist.loc[len(all_dist),:] = [cloc_name,j+1,'Stim',stim_range_per_dim[j]]
        all_dist.loc[len(all_dist),:] = [cloc_name,j+1,'Shuffle',spon_range_per_dim_s[j]]

#%% Plot all dist distributions.

plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (5,8),dpi = 180)
# ax.axhline(y = 0,color='gray', linestyle='--')
sns.barplot(data = all_dist,x = 'Var',y = 'Dim',hue = 'DataType',ax = ax,width = 0.5,orient="y")
# ax.set_title('Stim-like Ensemble Repeat Similarity',size = 10)
# ax.set_xlabel('')
# ax.set_ylabel('Pearson R')
ax.set_title('Dimension Variations dF/F')
# ax.set_xticklabels(['Real Data','Random Select'],size = 8)
plt.show()
