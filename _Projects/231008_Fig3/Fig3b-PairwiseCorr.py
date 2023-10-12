'''
This is the pairwise correlation fucntion.
But first, we need to get the relationship of different network, 
And add a step 0 of graph fit.

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
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from Kill_Cache import kill_all_cache
from sklearn.model_selection import cross_val_score
from sklearn import svm
import umap
import umap.plot
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import colorsys
import matplotlib as mpl

work_path = r'D:\_Path_For_Figs\Fig3_Cell_Seperation'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Datas_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
def Mises_Function(c_angle,best_angle,a0,b1,b2,c1,c2):
    '''
    Basic orientation fit function. Angle need to be input as RADIUS!!!!
    Parameters see the elife essay.
    '''
    y = a0+b1*np.exp(c1*np.cos(c_angle-best_angle))+b2*np.exp(c2*np.cos(c_angle-best_angle-np.pi))
    return y
#%%########################################################
# first we get all cell infos, includding a fit to all stim cells.
All_Cell_Frames = pd.DataFrame(index=list(range(1500000)),columns=['Loc','Cell_Name','Xloc','Yloc','OD_t','OD_index','Best_Orien','Fitted_Best_Orien','Best_Color','Orien_Fit_R2'])
fit_thres = 0.7

for i,c_loc in tqdm(enumerate(all_path_dic)):

    c_loc_name = c_loc.split('\\')[-1]
    c_ac = ot.Load_Variable_v2(c_loc,'Cell_Class.pkl')
    ## Fit graphs here,
    c_orien_info = c_ac.Orien_t_graphs
    orien_mat = np.zeros(shape = (len(c_ac.acn),8))
    cc_list = c_ac.acn
    orien_mat[:,0] = np.array(c_orien_info['Orien0-0'].loc['A_reponse'])
    orien_mat[:,1] = np.array(c_orien_info['Orien22.5-0'].loc['A_reponse'])
    orien_mat[:,2] = np.array(c_orien_info['Orien45-0'].loc['A_reponse'])
    orien_mat[:,3] = np.array(c_orien_info['Orien67.5-0'].loc['A_reponse'])
    orien_mat[:,4] = np.array(c_orien_info['Orien90-0'].loc['A_reponse'])
    orien_mat[:,5] = np.array(c_orien_info['Orien112.5-0'].loc['A_reponse'])
    orien_mat[:,6] = np.array(c_orien_info['Orien135-0'].loc['A_reponse'])
    orien_mat[:,7] = np.array(c_orien_info['Orien157.5-0'].loc['A_reponse'])
    angle_rad = np.arange(0,180,22.5)*np.pi/180
    # fit best tunings.
    best_oriens_fit = np.zeros(len(c_ac.acn))
    good_fit_num = 0
    for j,cc in tqdm(enumerate(c_ac.acn)):
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
        # fit to get best angle, if fit fail, return 
        if np.isnan(parameters).sum() == 0:
            filled_angle = np.arange(0,2*np.pi,0.01)
            pred_y = Mises_Function(filled_angle,parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5])
            # calculate best angle 
            best_angle_loc = np.where(pred_y == pred_y.max())[0][0]
            best_angle_rad = filled_angle[best_angle_loc]
            best_angle = best_angle_rad*180/np.pi
            # estimate r, if fail, use origional best orien.
            pred_y_r2 = Mises_Function(angle_rad,parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5])
            r2 = r2_score(orien_mat[j,:],pred_y_r2)
        else:
            r2 = 0
        # fill fitted best orien.
        if r2>fit_thres:
            best_oriens_fit[j] = best_angle
            good_fit_num += 1
        elif c_ac.all_cell_tunings[cc]['Best_Orien'] != 'False':
            best_oriens_fit[j] = float(c_ac.all_cell_tunings[cc]['Best_Orien'][5:])
        else:
            best_oriens_fit[j] = -1
        
        All_Cell_Frames.loc[len(All_Cell_Frames),:] = [c_loc_name,cc,c_ac.Cell_Locs[cc]['X'],c_ac.Cell_Locs[cc]['Y'],c_ac.all_cell_tunings[cc]['OD'],c_ac.all_cell_tunings[cc]['OD_index'],c_ac.all_cell_tunings[cc]['Best_Orien'],best_angle,c_ac.all_cell_tunings[cc]['Best_Color'],r2]
    good_fit_prop = good_fit_num/len(c_ac.acn)
    print(f'Loc {c_loc_name} have good fit rate {good_fit_prop*100:.2f} %')
ot.Save_Variable(work_path,'All_Cell_Tuning_Information',All_Cell_Frames)
del c_ac
#%%###################################################################
# Pairwise correlation calculation.
# select only R2>0.7's Neurons.
tuned_cell_frames = All_Cell_Frames[All_Cell_Frames['Orien_Fit_R2']>0.7]
# and limit best orietation to 0-180
for i in range(len(tuned_cell_frames)):
    if tuned_cell_frames.iloc[i,7]>180:
        tuned_cell_frames.iloc[i,7] = tuned_cell_frames.iloc[i,7]-180
# and calculate pairwise-corr frames.
cell_by_loc = tuned_cell_frames.groupby('Loc')
Pair_Corr_Frame = pd.DataFrame(index = list(range(1500000)),columns=['Loc','Dist','OD_Diff','OD_t_Diff','Orien_Diff','Cell_A','Cell_B','PearsonR','SpearmanR'])# avoid redefine, acelerate calculation.
counter = 0
for i,c_loc in enumerate(all_path_dic):
    c_loc_name = c_loc.split('\\')[-1]
    c_spon_frame = ot.Load_Variable(c_loc,'Spon_Before.pkl')
    c_info_group = cell_by_loc.get_group(c_loc_name)
    c_info_group['OD_index'] = c_info_group['OD_index'].clip(-1,1) # clip this series range.
    for j in tqdm(range(len(c_info_group))):
        cell_A = c_info_group.iloc[j,:]
        cell_A_series = np.array(c_spon_frame[cell_A['Cell_Name']])
        for k in range(j+1,len(c_info_group)):
            cell_B = c_info_group.iloc[k,:]
            cell_B_series = np.array(c_spon_frame[cell_B['Cell_Name']])
            r_pear,_ = stats.pearsonr(cell_A_series,cell_B_series)
            r_spear,_ = stats.spearmanr(cell_A_series,cell_B_series)
            dist = np.sqrt((cell_A['Xloc']-cell_B['Xloc'])**2+(cell_A['Yloc']-cell_B['Yloc'])**2)
            od_diff = np.abs(cell_A['OD_index']-cell_B['OD_index'])
            od_diff_t = np.abs(cell_A['OD_t']-cell_B['OD_t'])
            orien_diff = abs(cell_A['Fitted_Best_Orien']-cell_B['Fitted_Best_Orien'])
            orien_diff = min(orien_diff,180-orien_diff)
            Pair_Corr_Frame.loc[counter,:] = [c_loc_name,dist,od_diff,od_diff_t,orien_diff,cell_A['Cell_Name'],cell_B['Cell_Name'],r_pear,r_spear]
            counter +=1
# drop extra parts of the list.
Pair_Corr_Frame = Pair_Corr_Frame.dropna(axis=0)
ot.Save_Variable(work_path,'Pairwise_Corr',Pair_Corr_Frame)
#%%#################################################
#  After plot, we can generate correlation maps.
# KDE plot will cost almost 1h, be aware.
Pair_Corr_Frame['OD_Diff'] = Pair_Corr_Frame['OD_Diff'].clip(0,2)

plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,10),dpi = 180)
frame = Pair_Corr_Frame.loc[:,['Dist','OD_Diff','Orien_Diff']].convert_dtypes() # as float 64
frame = frame.sample(10000)
pd.plotting.scatter_matrix(frame,ax = axes) # this graph is ugly. We plot graphs by hand.
# 
# axes[i,j].set_ylim(0.0,1.0)
# sns.kdeplot(data=Pair_Corr_Frame, x="Dist", y="OD_t_Diff",fill=True, thresh=0, levels=100, cmap="hot",ax = axes[0,1])
# sns.kdeplot(data=Pair_Corr_Frame, x="Dist", y="Orien_Diff",fill=True, thresh=0, levels=100, cmap="hot",ax = axes[0,2])
# sns.kdeplot(data=Pair_Corr_Frame, x="OD_t_Diff", y="Dist",fill=True, thresh=0, levels=100, cmap="hot",ax = axes[1,0])
# sns.kdeplot(data=Pair_Corr_Frame, x="OD_t_Diff", y="Orien_Diff",fill=True, thresh=0, levels=100, cmap="hot",ax = axes[1,2])
# sns.kdeplot(data=Pair_Corr_Frame, x="Orien_Diff", y="Dist",fill=True, thresh=0, levels=100, cmap="hot",ax = axes[2,0])
# sns.kdeplot(data=Pair_Corr_Frame, x="Orien_Diff", y="OD_t_Diff",fill=True, thresh=0, levels=100, cmap="hot",ax = axes[2,1])
fig.suptitle('Cell-by-Cell Tuning Difference',fontsize = 20)
fig.tight_layout()
plt.show()

#%% Do Regression model of full model, and getting explained VAR of all Parameters.
# def Full_Model(X_Array,a,c,d,e):# full model of graph fitting.
#     # y = a*np.exp(-X_Array[0]*b)+c*X_Array[1]+d*X_Array[2]+e
#     y = a*X_Array[0]+c*X_Array[1]+d*X_Array[2]+e
#     return y


# all_loc = list(set(Pair_Corr_Frame['Loc']))
# loc_corrs = Pair_Corr_Frame.groupby('Loc')
# r2_list = []
# for i,c_loc in enumerate(all_loc):
#     c_group = loc_corrs.get_group(c_loc)
#     c_dist = np.array(c_group['Dist'])
#     c_od = np.array(c_group['OD_Diff'])
#     c_orien = np.array(c_group['Orien_Diff'])
#     c_corr = np.array(c_group['PearsonR'])
#     c_X_array = np.array([c_dist,c_od,c_orien])

#     parameters, covariance = curve_fit(Full_Model, c_X_array,c_corr,maxfev=30000,p0=[-0.05,-0.1,-0.05,0.4])
#     predicted_corr =np.zeros(c_X_array.shape[1])
#     for j in tqdm(range(len(predicted_corr))):
#         predicted_corr[j] = Full_Model(c_X_array[:,j],parameters[0],parameters[1],parameters[2],parameters[3])
#     r2 = r2_score(c_corr,predicted_corr)
#     r2_list.append(r2)
#     c_group['Corr_Predicted'] = predicted_corr
#%% SKLearn linear regression.
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
regression_frame = pd.DataFrame(columns=['Loc','Global_Corr','Dist_Para','OD_Para','Orien_Para','Dist_VAR','OD_VAR','Orien_VAR','FittedR2'])

for i,c_loc in enumerate(all_loc):
    c_loc_name = c_loc.split('\\')[-1]
    c_group = loc_corrs.get_group(c_loc)
    c_dist = np.array(c_group['Dist'])
    c_od = np.array(c_group['OD_Diff'])
    c_orien = np.array(c_group['Orien_Diff'])
    c_corr = np.array(c_group['PearsonR'])
    c_X_array = np.array([c_dist,c_od,c_orien])
    model = LinearRegression()
    model.fit(c_X_array.T, c_corr)
    intercept = model.intercept_
    parameters = model.coef_
    y_pred = model.predict(c_X_array.T)
    r2 = r2_score(c_corr,y_pred)
    partial_r2 = []
    for j in range(c_X_array.T.shape[1]):
        X_i = np.delete(c_X_array.T, j, axis=1)  # Remove the i-th variable from X
        model_i = LinearRegression()
        model_i.fit(X_i, c_corr)
        y_pred_i = model_i.predict(X_i)
        explained_variance_i = r2_score(c_corr, y_pred_i)
        # partial_r2.append(r2-explained_variance_i)
        partial_r2.append(explained_variance_i)
    
    regression_frame.loc[len(regression_frame),:] = [c_loc_name,intercept,parameters[0],parameters[1],parameters[2],partial_r2[0],partial_r2[1],partial_r2[2],r2]
#%% Plot linear regression result.
template_loc = all_loc[2]
c_loc_name = c_loc.split('\\')[-1]
c_group = loc_corrs.get_group(c_loc)
c_dist = np.array(c_group['Dist'])
c_od = np.array(c_group['OD_Diff'])
c_orien = np.array(c_group['Orien_Diff'])
c_corr = np.array(c_group['PearsonR'])
c_X_array = np.array([c_dist,c_od,c_orien])
model = LinearRegression()
model.fit(c_X_array.T, c_corr)
intercept = model.intercept_
parameters = model.coef_
predicted_corr = model.predict(c_X_array.T)
c_group['Predicted_Corr'] = predicted_corr
#%%
plt.clf()
plt.cla()
selected_points = c_group.sample(20000)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,4),dpi = 180)
sns.kdeplot(data=selected_points, x="Dist", y="PearsonR",fill=True,levels=100,thresh=0,  cmap="hot",ax = axes[0])
sns.kdeplot(data=selected_points, x="OD_Diff", y="PearsonR",fill=True,thresh=0, levels=100, cmap="hot",ax = axes[1])
sns.kdeplot(data=selected_points, x="Orien_Diff", y="PearsonR",fill=True, thresh=0, levels=100, cmap="hot",ax = axes[2])
# sns.scatterplot(data=selected_points, x="Dist", y="Predicted_Corr",color = '#39C5BB',ax = axes[0],s = 3,alpha = 0.7)
# sns.scatterplot(data=selected_points, x="OD_Diff", y="Predicted_Corr",color = '#39C5BB',ax = axes[1],s = 3,alpha = 0.7)
# sns.scatterplot(data=selected_points, x="Orien_Diff", y="Predicted_Corr",color = '#39C5BB',ax = axes[2],s = 3,alpha = 0.7)
selected_points['Predicted_Corr'] = selected_points['Predicted_Corr'].astype('f8')
selected_points['Dist'] = selected_points['Dist'].astype('f8')
selected_points['OD_Diff'] = selected_points['OD_Diff'].astype('f8')
selected_points['Orien_Diff'] = selected_points['Orien_Diff'].astype('f8')
sns.regplot(data=selected_points, x="Dist", y="Predicted_Corr",color='#39C5BB',ax = axes[0],scatter = False,ci = 99,scatter_kws={'alpha':0.7})
sns.regplot(data=selected_points, x="OD_Diff", y="Predicted_Corr",color='#39C5BB',ax = axes[1],scatter = False,ci = 99,scatter_kws={'alpha':0.7})
sns.regplot(data=selected_points, x="Orien_Diff", y="Predicted_Corr",color='#39C5BB',ax = axes[2],scatter = False,ci = 99,scatter_kws={'alpha':0.7})

axes[0].set(ylim = (0,0.7))
axes[1].set(ylim = (0,0.7))
axes[2].set(ylim = (0,0.7))
axes[0].set(xlim = (0,600))
axes[1].set(xlim = (0,2))
axes[2].set(xlim = (0,90))
axes[1].yaxis.set_visible(False)
axes[2].yaxis.set_visible(False)

fig.suptitle('Correlation with Tuning Difference',fontsize = 20)
fig.tight_layout()
plt.show()