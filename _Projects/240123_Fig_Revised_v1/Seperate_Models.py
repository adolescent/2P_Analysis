'''
Generate seperated G16 only,OD only, Orien only umap classifier, avoid random diff.

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

all_path_dic_v2 = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V2'))

import warnings
warnings.filterwarnings("ignore")
#%% ############################### 1. GENERATE ALL V1 MODELS################################
for i,cloc in tqdm(enumerate(all_path_dic)):
    c_ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    # Generate G16 Model
    g16_frames,g16_ids = c_ac.Combine_Frame_Labels(od = False,color = False)
    model_orien = umap.UMAP(n_components=3,n_neighbors=20)
    model_orien.fit(g16_frames)
    ot.Save_Variable(cloc,'Orien_UMAP_3D_20comp',model_orien)
    # Generate OD Model
    od_frames,od_ids = c_ac.Combine_Frame_Labels(od = True,orien = False,color = False)
    model_od = umap.UMAP(n_components=3,n_neighbors=20)
    model_od.fit(od_frames)
    ot.Save_Variable(cloc,'OD_UMAP_3D_20comp',model_od)
    # Generate Hue Model
    hue_frames,hue_ids = c_ac.Combine_Frame_Labels(od = False,orien = False,color = True)
    model_hue = umap.UMAP(n_components=3,n_neighbors=20)
    model_hue.fit(hue_frames)
    ot.Save_Variable(cloc,'Color_UMAP_3D_20comp',model_hue)

#%% ############################# 2. DO THE SAME ON V2############################
for i,cloc in tqdm(enumerate(all_path_dic_v2)):
    c_ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    # Generate G16 Model
    g16_frames,g16_ids = c_ac.Combine_Frame_Labels(od = False,color = False)
    model_orien = umap.UMAP(n_components=3,n_neighbors=20)
    model_orien.fit(g16_frames)
    ot.Save_Variable(cloc,'Orien_UMAP_3D_20comp',model_orien)
    # Generate OD Model
    if i != 0: # loc 0 have no od.
        od_frames,od_ids = c_ac.Combine_Frame_Labels(od = True,orien = False,color = False)
        model_od = umap.UMAP(n_components=3,n_neighbors=20)
        model_od.fit(od_frames)
        ot.Save_Variable(cloc,'OD_UMAP_3D_20comp',model_od)
    # Generate Hue Model
    hue_frames,hue_ids = c_ac.Combine_Frame_Labels(od = False,orien = False,color = True)
    model_hue = umap.UMAP(n_components=3,n_neighbors=20)
    model_hue.fit(hue_frames)
    ot.Save_Variable(cloc,'Color_UMAP_3D_20comp',model_hue)
