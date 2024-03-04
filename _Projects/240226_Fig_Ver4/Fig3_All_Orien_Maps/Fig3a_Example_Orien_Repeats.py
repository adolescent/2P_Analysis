'''
This file shows an example of orien corr's cos similarity.
Loc L76-18M Used.

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

work_path = r'D:\_Path_For_Figs\240228_Figs_v4\Fig3'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
# some times we need to ignore warnings.
import warnings
warnings.filterwarnings("ignore")

example_loc = r'D:\_All_Spon_Data_V1\L76_18M_220902'

#%% ################# 0.Define Basic Functions.#################
def Find_Example(corr_mat,spon_label,c_spon,center = 30,width = 3,min_corr =0.5):
    find_from = corr_mat[corr_mat.min(1)<min_corr]
    best_locs = find_from.idxmax(1)
    satistied_series = np.where((best_locs>(center-width))*(best_locs<(center+width)))[0]
    # best_id = Corr_Matrix_Norm.loc[satistied_series,:].max(1).idxmax()
    best_id = find_from.iloc[satistied_series,:].max(1).idxmax()
    origin_class = spon_label[best_id]
    origin_frame = ac.Generate_Weighted_Cell(c_spon.iloc[best_id,:])
    corr_series = corr_mat.loc[best_id,:]
    best_orien = corr_series.idxmax()
    best_corr = corr_series.max()
    print(f'Best Orientation {best_orien}, with corr {best_corr}.')
    print(f'UMAP Classified Class:{origin_class}')
    return origin_frame,origin_class,corr_series,best_orien,best_corr

#%% ###################### 1. Get All Orien Corrs of Example Locs.#################################
ac = ot.Load_Variable_v2(example_loc,'Cell_Class.pkl')
