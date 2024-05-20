'''
This graph generate Pairwise Correlation data of all data points.

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
from Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *

work_path = r'D:\_Path_For_Figs\240228_Figs_v4\Fig5'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
# some times we need to ignore warnings.
import warnings
warnings.filterwarnings("ignore")



#%% ################ 1. GENERATE PAIR CORR MATRIX.###################
all_best_oriens = ot.Load_Variable(r'D:\_Path_For_Figs\240228_Figs_v4\Fig3\VAR2_All_Cell_Best_Oriens.pkl')
all_cell_corr = {}
for i,cloc in enumerate(all_path_dic): # test 1 location.
    cloc_name = cloc.split('\\')[-1]
    c_best_orien = all_best_oriens[cloc_name]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    c_tuned_cells_orien = c_best_orien[c_best_orien['Tuned']==1]
    c_tuned_cells = list(c_tuned_cells_orien.index)
    pairnum = int(len(c_tuned_cells)*(len(c_tuned_cells)-1)/2)
    cloc_corr_frame = pd.DataFrame(0,range(pairnum),columns = ['Corr','CellA','CellB','DistX','DistY','OD_A','OD_B','OrienA','OrienB','Dist','OD_Diff','Orien_Diff'])
    counter = 0
    cloc_OD = ac.OD_t_graphs['OD'].loc['CohenD']
    for j in tqdm(range(len(c_tuned_cells))):
        cell_A = c_tuned_cells[j]
        cell_A_coords = ac.Cell_Locs[cell_A]
        spon_A = np.array(c_spon.loc[:,cell_A])
        od_A = cloc_OD[cell_A]
        best_orien_A = c_tuned_cells_orien.loc[cell_A,'Best_Angle']
        for k in range(j+1,len(c_tuned_cells)):
            cell_B = c_tuned_cells[k]
            cell_B_coords = ac.Cell_Locs[cell_B]
            spon_B = np.array(c_spon.loc[:,cell_B])
            od_B = cloc_OD[cell_B]
            best_orien_B = c_tuned_cells_orien.loc[cell_B,'Best_Angle']
            # calculate difference,
            c_corr,_ = stats.pearsonr(spon_A,spon_B)
            c_distx = cell_A_coords['X']-cell_B_coords['X']
            c_disty = cell_A_coords['Y']-cell_B_coords['Y']
            c_od_diff = abs(od_A-od_B)
            c_dist = np.sqrt(c_distx**2+c_disty**2)
            c_orien_diff = abs(best_orien_A-best_orien_B)
            c_orien_diff = min(c_orien_diff,180-c_orien_diff)
            cloc_corr_frame.loc[counter,:] = [c_corr,cell_A,cell_B,c_distx,c_disty,od_A,od_B,best_orien_A,best_orien_B,c_dist,c_od_diff,c_orien_diff]
            counter += 1
    all_cell_corr[cloc_name] = cloc_corr_frame
ot.Save_Variable(work_path,'All_Pair_Corrs',all_cell_corr)


