'''
This script generate all orientation maps on all data points.

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
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
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
ot.Save_Variable(work_path,'All_Orien_Response',all_orien_response)
ot.Save_Variable(work_path,'All_Cell_Best_Oriens',all_cell_best_oriens)

#%% ##################   FIG S3A- ALL LOC RESPONSE MAPS ##########################
from matplotlib.animation import FuncAnimation
all_loc_graph_path = ot.join(work_path,'All_Orientation_Maps')
ot.mkdir(all_loc_graph_path)
for i,cloc in tqdm(enumerate(all_path_dic[4:])):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_savepath = ot.join(all_loc_graph_path,cloc_name)
    ot.mkdir(c_savepath)
    c_responses = all_orien_response[cloc_name]
    # get all orientation graphs.
    frame_for_gif = np.zeros(shape = (512,512,180))
    for j in range(180):
        c_response = c_responses.loc[:,j]
        c_map = ac.Generate_Weighted_Cell(c_response)
        frame_for_gif[:,:,j] = c_map
        #and save current figure in folder.
        # plt.clf()
        # plt.cla()
        value_max = 3
        value_min = -1
        fig, ax = plt.subplots(figsize=(5,5),dpi = 180)
        ax.set_title(f'Orientation {j}')
        heatmap = sns.heatmap(c_map, center = 0, ax=ax,xticklabels=False,yticklabels=False,vmax = value_max,vmin = value_min,square=True,cbar=False)
        fig.tight_layout()
        fig.savefig(ot.join(c_savepath,f'Orientation_{j}.png'))
        plt.close('all')
    # plt.clf()
    # plt.cla()
    # Save gif Here.
    plt.close('all')
    fig, ax = plt.subplots(figsize=(5,5),dpi = 180)
    heatmap = sns.heatmap(frame_for_gif[:, :, 0], center = 0, ax=ax,xticklabels=False,yticklabels=False,vmax = value_max,vmin = value_min,square=True,cbar=False)
    def update(frame):
        ax.cla()  # Clear the axis
        heatmap = sns.heatmap(frame_for_gif[:, :, frame], center = 0, ax=ax,xticklabels=False,yticklabels=False,vmax = value_max,vmin = value_min,square=True,cbar=False)
        ax.set_title(f'Orientation {str(1000+frame)[1:]}')
        return heatmap
    animation = FuncAnimation(fig, update, frames=range(180), interval=120)
    animation.save(ot.join(c_savepath,f'_All_Orientation_GIF.gif'), writer='pillow')
    plt.close('all')


