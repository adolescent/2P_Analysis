'''
This script is stats version of Fig3b, and we will plot all response map's best orientation in Radar map.

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

work_path = r'D:\_Path_For_Figs\2401_Amendments\Fig3_New'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
# some times we need to ignore warnings.
import warnings
warnings.filterwarnings("ignore")
#%% 
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


#%% ########################## 1. GENERATE ALL ORIEN CORRS ############################

all_orien_response = ot.Load_Variable(work_path,'All_Orien_Response.pkl')
all_cell_best_oriens = ot.Load_Variable(work_path,'All_Cell_Best_Oriens.pkl')
all_loc_corr_matrix = {}
all_locnames = list(all_orien_response.keys())

for i,cloc in enumerate(all_path_dic):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    c_model = ot.Load_Variable(cloc,'All_Stim_UMAP_3D_20comp.pkl')
    analyzer = UMAP_Analyzer(ac = ac,umap_model=c_model,spon_frame=c_spon)
    analyzer.Train_SVM_Classifier()
    spon_label = analyzer.spon_label
    c_Corr_Matrix = pd.DataFrame(0.0,index=range(len(c_spon)),columns=range(180))
    c_orien_response = all_orien_response[cloc_name].clip(-5,5)
    for j in tqdm(range(len(c_spon))):
        single_spon = np.array(c_spon)[j,:]
        # calculate cosine similarity with all orientation maps.
        for k in range(180):
            c_orien_map = np.array(c_orien_response.iloc[:,k])
            cos_sim = single_spon.dot(c_orien_map) / (np.linalg.norm(single_spon) * np.linalg.norm(c_orien_map))
            c_Corr_Matrix.loc[j,k] = cos_sim
    # normalize by std.
    c_Corr_Matrix_Norm = copy.deepcopy(c_Corr_Matrix)
    for j in range(180):
        c_oren_disp = np.array(c_Corr_Matrix.iloc[:,j])
        c_Corr_Matrix_Norm.iloc[:,j] = (c_Corr_Matrix_Norm.iloc[:,j])/c_oren_disp.std()
    all_loc_corr_matrix[cloc_name] = c_Corr_Matrix_Norm
ot.Save_Variable(work_path,'All_Location_Corr_Matrix',all_loc_corr_matrix)
#%% Test plot 
# used_corr = c_Corr_Matrix_Norm
# used_corr['Best_Angle'] = used_corr.idxmax(1)
# sorted_mat = used_corr.sort_values(by=['Best_Angle'])
# sorted_mat = sorted_mat.drop(['Best_Angle'],axis = 1)
# plt.clf()
# plt.cla()
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4),dpi = 180)
# sns.heatmap(sorted_mat.iloc[:,:-1],center = 0,vmax = 2.5,vmin = -1,xticklabels=False,yticklabels=False,ax = ax)
# # Corr_Matrix_Norm = Corr_Matrix_Norm.drop(['Best_Angle'],axis = 1)
# ax.set_title('Similarity with All Orientation Maps')
# ax.set_ylabel('Frames')
# ax.set_xticks([0,45,90,135])
# ax.set_xticklabels([0,45,90,135])
# ax.set_xlabel('Orientation Angles')
# fig.tight_layout()
# plt.show()
c_Corr_Matrix_Norm = all_loc_corr_matrix[all_locnames[4]]
seperate_para = 1
c_Corr_Matrix_Norm = c_Corr_Matrix_Norm.astype('f8')
big_parts = c_Corr_Matrix_Norm[c_Corr_Matrix_Norm.max(1)>seperate_para]
small_parts = c_Corr_Matrix_Norm[c_Corr_Matrix_Norm.max(1)<seperate_para]
big_parts['Best_Angle'] = big_parts.idxmax(1)
small_parts['Best_Angle'] = small_parts.idxmax(1)
sorted_mat_a = big_parts.sort_values(by=['Best_Angle'])
sorted_mat_b = small_parts.sort_values(by=['Best_Angle'])
sorted_mat = pd.concat([sorted_mat_a,sorted_mat_b])
# sorted_mat['Max'] = sorted_mat.max(1)
# sorted_mat = sorted_mat.sort_values(by=['Max'])
# sorted_mat = sorted_mat.drop(['Best_Angle'],axis = 1)
# sorted_mat = sorted_mat.drop(['Max'],axis = 1)
plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4),dpi = 180)
sns.heatmap(sorted_mat.iloc[:,:-1],center = 0,vmax = 2.5,vmin = -1,xticklabels=False,yticklabels=False,ax = ax)
# Corr_Matrix_Norm = Corr_Matrix_Norm.drop(['Best_Angle'],axis = 1)
ax.set_title('Similarity with All Orientation Maps')
ax.set_ylabel('Frames')
ax.set_xticks([0,45,90,135])
ax.set_xticklabels([0,45,90,135])
ax.set_xlabel('Orientation Angles')

fig.tight_layout()
plt.show()
#%% ######################## 2.PLOT PLOAR HISTOGRAM##########################################
'''
This part will select all orientation repeats, and analyze their orientation preference.
'''
all_orien_matrix = pd.DataFrame(columns = ['Loc','Best_Orien','Strength','Tuning_Index'])
thres_max = 1.5 # biggest corr above this are correlated
thres_min = 10# smallest corr below this are not global.

for i,cloc_name in enumerate(all_locnames):
    cloc_corr = all_loc_corr_matrix[cloc_name]
    fitted_corrs = cloc_corr[cloc_corr.max(1)>thres_max]
    fitted_corrs = fitted_corrs[fitted_corrs.min(1)<thres_min]
    all_used_index = list(fitted_corrs.index)
    for j,c_index in tqdm(enumerate(all_used_index)):
        c_curve = fitted_corrs.loc[c_index,:]
        c_max_angle = c_curve.idxmax()
        c_max_corr = c_curve.max()
        c_min_corr =  c_curve.min()
        c_tune = (c_max_corr-c_min_corr)/(c_max_corr+c_min_corr)
        all_orien_matrix.loc[len(all_orien_matrix),:] = [cloc_name,c_max_angle,c_max_corr,c_tune]

#%%
plt.clf()
plt.cla()
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
n_bins = 30
rads = np.radians(np.array(all_orien_matrix['Best_Orien'].astype('f8')))*2

ax.set_xticks(np.arange(0, 2*np.pi, 2*np.pi/6))
ax.set_xticklabels(['0', '30', '60', '90', '120', '150'])
ax.set_rlim(0,400)
ax.set_rticks([0,100,200,300,400])
ax.set_rlabel_position(45) 
# ax.set_xlabel('Repeat Counts')
ax.hist(rads, bins=n_bins,rwidth=1)
ax.set_title('All Orientation Repeat in Spontaneous')
fig.tight_layout()
plt.show()
#%% stats info of all data points.
all_orien_repeat_dic = dict(list(all_orien_matrix.groupby('Loc')))
all_counts = np.zeros(8)
for i,cloc in enumerate(all_orien_repeat_dic):
    c_spon_len = len(all_loc_corr_matrix[cloc])
    c_frame = len(all_orien_repeat_dic[cloc])
    all_counts[i]=(c_frame/c_spon_len)