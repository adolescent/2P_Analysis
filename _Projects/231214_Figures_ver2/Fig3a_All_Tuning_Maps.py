'''
This script will calculate all tuning maps for all locations, and generate 

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

work_path = r'D:\_Path_For_Figs\_2312_ver2\Fig2'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)

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
    pass