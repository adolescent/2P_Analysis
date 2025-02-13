'''
This is a big one, start from:
1. Getting orientation tuning information for cells in V2
2. For orientation tuned cell, calculate it's pairwise corr info.

V2 may not having od, so od part here is set as -1.
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

all_path_dic = list(ot.Get_Subfolders(r'D:\_DataTemp\_Fig_Datas\_All_Spon_Data_V2'))

# all_path_dic.pop(4)
# all_path_dic.pop(6)
save_path = r'G:\我的云端硬盘\#Figs\#250211_Revision1\Fig5'
#%%
'''
Part 1, fit V2 cell for Orientation tuning info.
'''
#%%########################## 0. BASIC FUNCTIONS############
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
def Mises_Function(c_angle,best_angle,a0,b1,b2,c1,c2):
    '''
    Basic orientation fit function. Angle need to be input as RADIUS!!!!
    Parameters see the elife essay.
    '''
    y = a0+b1*np.exp(c1*np.cos(c_angle-best_angle))+b2*np.exp(c2*np.cos(c_angle-best_angle-np.pi))
    return y

def Fit_Mises(ac,fit_thres = 0.7,used_frame = [4,5]):
    orien_cr_response = ac.orien_CR_Response
    orien_mat = np.zeros(shape = (len(ac.acn),16))
    for i,cc in enumerate(ac.acn):
        cc_cr = orien_cr_response[cc]
        for j in range(16):
            cc_cond_response = cc_cr[j+1]
            orien_mat[i,j] = cc_cond_response[used_frame].mean()
    angle_rad = np.arange(0,360,22.5)*np.pi/180
    all_fitting_para_dic = {}
    good_fit_num = 0
    for j,cc in tqdm(enumerate(ac.acn)):
    # cc_best_orien_response = c_ac.all_cell_tunings[cc]['Best_Orien'][5:]
        try:
            parameters, covariance = curve_fit(Mises_Function, angle_rad,orien_mat[j,:],maxfev=30000)
        except RuntimeError:
            try:
                parameters, covariance = curve_fit(Mises_Function, angle_rad,orien_mat[j,:],maxfev=50000,p0 = [0,0,0,0,0,0])
            except RuntimeError:
                try:
                    parameters, covariance = curve_fit(Mises_Function, angle_rad,orien_mat[j,:],maxfev=50000,p0 = [0.2,1,-0.4,-0.2,1,1.5])
                except RuntimeError:
                    parameters = np.array([np.nan])
        if len(parameters) == 6: # Second check for good fit.
            pred_y_r2 = Mises_Function(angle_rad,parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5])
            r2 = r2_score(orien_mat[j,:],pred_y_r2)
            if r2<fit_thres:
                parameters = np.array([np.nan])
            else:
                good_fit_num +=1
        if len(parameters) != 6:# Use mean response to replace fit here.
            parameters = np.array([orien_mat[j,:].mean()])
        all_fitting_para_dic[cc] = parameters
    print(f'Good Fitted Cell Number: {good_fit_num} ({(good_fit_num/len(ac.acn))*100:.1f}%)')
    return all_fitting_para_dic,good_fit_num

def Generate_Single_Orien_Response(degree,all_fitting_para_dic):
    acn = list(all_fitting_para_dic.keys())
    angle = degree*np.pi/180
    orien_map = np.zeros(len(acn))
    for j,cc in enumerate(acn):
        cc_para = all_fitting_para_dic[cc]
        if len(cc_para) == 1: # use average 
            c_response = cc_para[0]
        else:
            c_response1 = Mises_Function(angle,cc_para[0],cc_para[1],cc_para[2],cc_para[3],cc_para[4],cc_para[5])
            c_response2 = Mises_Function(angle+np.pi,cc_para[0],cc_para[1],cc_para[2],cc_para[3],cc_para[4],cc_para[5])
            # c_response = np.clip(c_response,-1,3)
            c_response = (c_response1+c_response2)/2
        orien_map[j] = c_response
    return orien_map


#%%##############################1. GENERATE ALL ORIENTATION MAPS###############################
all_orien_response = {}
all_cell_best_oriens = {}
for i,cloc in enumerate(all_path_dic):
    print(f'Processing Location {i+1}')
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    # c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    c_ac_fitting_para,c_good_fits = Fit_Mises(ac)
    # save response of all orientation and tuning of all cells.
    all_orien_response[cloc_name] = pd.DataFrame(0,index = ac.acn,columns = range(0,180))
    all_cell_best_oriens[cloc_name] = pd.DataFrame(0,index = ac.acn,columns = ['Cell','Best_Angle','Tuned'])
    # cycle all cells.
    for j,cc in enumerate(ac.acn):
        cc_para = c_ac_fitting_para[cc]
        prediction_angle = np.linspace(0,np.pi,180)
        if len(cc_para) == 1: # use average 
            c_response = cc_para[0]
            all_cell_best_oriens[cloc_name].loc[cc,:] = [cc,-1,False]
            all_orien_response[cloc_name].loc[cc,:] = c_response 
        else:
            c_response1 = Mises_Function(prediction_angle,cc_para[0],cc_para[1],cc_para[2],cc_para[3],cc_para[4],cc_para[5])
            c_response2 = Mises_Function(prediction_angle+np.pi,cc_para[0],cc_para[1],cc_para[2],cc_para[3],cc_para[4],cc_para[5])
            # c_response = np.clip(c_response,-1,3)
            c_response = (c_response1+c_response2)/2
            c_best_angle = c_response.argmax() # as for 1 degree, we can use id directly. 
            all_cell_best_oriens[cloc_name].loc[cc,:] = [cc,c_best_angle,True]
            all_orien_response[cloc_name].loc[cc,:] = c_response 
# save orientation fittings.
# ot.Save_Variable(work_path,'All_Orien_Response',all_orien_response)
ot.Save_Variable(save_path,'All_Cell_Best_Oriens_V2',all_cell_best_oriens)


#%%
'''
Part 2, getting pairwise corr of V2 cells.
'''
all_best_oriens = ot.Load_Variable(r'G:\我的云端硬盘\#Figs\#250211_Revision1\Fig5\All_Cell_Best_Oriens_V2.pkl')
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
    # cloc_OD = ac.OD_t_graphs['OD'].loc['CohenD']
    for j in tqdm(range(len(c_tuned_cells))):
        cell_A = c_tuned_cells[j]
        cell_A_coords = ac.Cell_Locs[cell_A]
        spon_A = np.array(c_spon.loc[:,cell_A])
        od_A = -1
        best_orien_A = c_tuned_cells_orien.loc[cell_A,'Best_Angle']
        for k in range(j+1,len(c_tuned_cells)):
            cell_B = c_tuned_cells[k]
            cell_B_coords = ac.Cell_Locs[cell_B]
            spon_B = np.array(c_spon.loc[:,cell_B])
            od_B = -1
            best_orien_B = c_tuned_cells_orien.loc[cell_B,'Best_Angle']
            # calculate difference,
            c_corr,_ = stats.pearsonr(spon_A,spon_B)
            c_distx = cell_A_coords['X']-cell_B_coords['X']
            c_disty = cell_A_coords['Y']-cell_B_coords['Y']
            c_od_diff = -1
            c_dist = np.sqrt(c_distx**2+c_disty**2)
            c_orien_diff = abs(best_orien_A-best_orien_B)
            c_orien_diff = min(c_orien_diff,180-c_orien_diff)
            cloc_corr_frame.loc[counter,:] = [c_corr,cell_A,cell_B,c_distx,c_disty,od_A,od_B,best_orien_A,best_orien_B,c_dist,c_od_diff,c_orien_diff]
            counter += 1
    all_cell_corr[cloc_name] = cloc_corr_frame
ot.Save_Variable(save_path,'All_Pair_Corrs_V2',all_cell_corr)