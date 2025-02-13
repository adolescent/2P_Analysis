'''
This script will do pairwise correlation for all cell pair in V1, including it's 

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
from sklearn.model_selection import cross_val_score
from sklearn import svm
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
from Cell_Class.Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *
from Review_Fix_Funcs import *
from Filters import Signal_Filter_v2
import warnings

warnings.filterwarnings("ignore")

all_path_dic = list(ot.Get_Subfolders(r'D:\_DataTemp\_Fig_Datas\_All_Spon_Data_V1'))

all_path_dic.pop(4)
all_path_dic.pop(6)
save_path = r'G:\我的云端硬盘\#Figs\#240802_Figs_Ver_CR&Elife\#Figs\Fig4'

#%% ################ 1. GENERATE PAIR CORR MATRIX.###################
all_best_oriens = ot.Load_Variable(r'G:\我的云端硬盘\#Figs\#250211_Revision1\Fig4\All_Cell_Best_Oriens.pkl')
all_cell_corr = {}
for i,cloc in enumerate(all_path_dic): # test 1 location.
    cloc_name = cloc.split('\\')[-1]
    c_best_orien = all_best_oriens[cloc_name]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    start = c_spon.index[0]
    end = c_spon.index[-1]
    c_spon = Z_refilter(ac,'1-001',start,end).T
    # transfer c_spon into pd frame
    c_spon = pd.DataFrame(c_spon,columns=ac.acn,index=range(len(c_spon)))

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
ot.Save_Variable(save_path,'All_Pair_Corrs',all_cell_corr)