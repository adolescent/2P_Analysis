'''
This script will try to get all orientation funcional maps and get recovery graph.
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
from Filters import Signal_Filter

work_path = r'D:\_Path_For_Figs\FigS2e_All_Orientation_Funcmap'
expt_folder = r'D:\_All_Spon_Datas_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
ac.wp = expt_folder
# Get stim label and stim response.
all_stim_frame,all_stim_label = ac.Combine_Frame_Labels(od = True,orien = True,color = True,isi = True)
spon_series = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
#%%############################# TUNING FIT ################################ 
#  Fit all cells and save fitting parameters.
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
def Mises_Function(c_angle,best_angle,a0,b1,b2,c1,c2):
    '''
    Basic orientation fit function. Angle need to be input as RADIUS!!!!
    Parameters see the elife essay.
    '''
    y = a0+b1*np.exp(c1*np.cos(c_angle-best_angle))+b2*np.exp(c2*np.cos(c_angle-best_angle-np.pi))
    return y
fit_thres = 0.7
orien_cr_response = ac.orien_CR_Response
used_frame = [4,5]
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
ot.Save_Variable(work_path,'Orientation_Fitting_Parameter',all_fitting_para_dic)
#%%###################### GET ALL ORIENTATION MAP #####################################
# 1.Generate all predicted response frame.
partnum = 180
prediction_angle = np.linspace(0,np.pi,partnum+1)
all_Orientation_Maps = pd.DataFrame(0,columns=ac.acn,index=range(partnum))
for i in tqdm(range(partnum)):
    for j,cc in enumerate(ac.acn):
        cc_para = all_fitting_para_dic[cc]
        if len(cc_para) == 1: # use average 
            c_response = cc_para[0]
        else:
            c_response1 = Mises_Function(prediction_angle[i],cc_para[0],cc_para[1],cc_para[2],cc_para[3],cc_para[4],cc_para[5])
            c_response2 = Mises_Function(prediction_angle[i]+np.pi,cc_para[0],cc_para[1],cc_para[2],cc_para[3],cc_para[4],cc_para[5])
            # c_response = np.clip(c_response,-1,3)
            c_response = (c_response1+c_response2)/2
        all_Orientation_Maps.loc[i,cc] = c_response
#%%###################### GENERATE 180 ORIEN PLOTS  ####################################
value_max = 3
value_min = -1
all_orien_path = ot.join(work_path,'All_Orien_Map')
ot.mkdir(all_orien_path)
for i in tqdm(range(partnum)):
    cc_response = all_Orientation_Maps.loc[i,:]
    cc_map = ac.Generate_Weighted_Cell(cc_response)
    plt.clf()
    plt.cla()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6,5),dpi = 180)
    cbar_ax = fig.add_axes([.89, .25, .02, .5])
    sns.heatmap(cc_map,center = 0,xticklabels=False,yticklabels=False,ax = axes,vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    fig.suptitle(f'Orientation {str(1000+i)[1:]} Prediction Map')
    fig.tight_layout()
    fig.savefig(f'{all_orien_path}\\Orien{str(1000+i)[1:]}.png')
    plt.close()
#%%########################### PLOT GIF ####################################
from matplotlib.animation import FuncAnimation
frame_for_gif = np.zeros(shape = (512,512,180))
for i in range(180):
    c_response = all_Orientation_Maps.loc[i,:]
    c_map = ac.Generate_Weighted_Cell(c_response)
    frame_for_gif[:,:,i] = c_map

plt.clf()
plt.cla()
fig, ax = plt.subplots(figsize=(5,5),dpi = 180)
heatmap = sns.heatmap(frame_for_gif[:, :, 0], center = 0, ax=ax,xticklabels=False,yticklabels=False,vmax = value_max,vmin = value_min,square=True,cbar=False)
def update(frame):
    ax.cla()  # Clear the axis
    heatmap = sns.heatmap(frame_for_gif[:, :, frame], center = 0, ax=ax,xticklabels=False,yticklabels=False,vmax = value_max,vmin = value_min,square=True,cbar=False)
    ax.set_title(f'Orientation {str(1000+frame)[1:]}')
    return heatmap

animation = FuncAnimation(fig, update, frames=range(180), interval=120)
animation.save(f'{work_path}\\All_Orientation.gif', writer='pillow')
plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% COMPARE ALL REPEATS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% step1, get embedding results
reducer = ot.Load_Variable(expt_folder,'All_Stim_UMAP_3D_20comp.pkl')
stim_embeddings = reducer.transform(all_stim_frame)
spon_embeddings = reducer.transform(spon_series)
classifier,score = SVM_Classifier(embeddings=stim_embeddings,label = all_stim_label)
predicted_spon_label = SVC_Fit(classifier,data = spon_embeddings,thres_prob = 0)
#%% step2, get all orien matrix and find corr.
all_orien_label = np.where((predicted_spon_label>8)*(predicted_spon_label<17))[0]
all_orien_frame = np.array(spon_series.iloc[all_orien_label,:])
umap_predicted_label = predicted_spon_label[all_orien_label]
Corr_Matrix = pd.DataFrame(0.0,index=range(len(all_orien_label)),columns=range(180))
# cycle all frame to get correlation matrix.
for i,c_graph in tqdm(enumerate(all_orien_frame)):
    single_orien = all_orien_frame[i,:]
    for j in range(180):
        c_orien_map = np.array(all_Orientation_Maps.iloc[j,:])
        c_r,_ = stats.pearsonr(single_orien,c_orien_map)
        Corr_Matrix.loc[i,j] = c_r
ot.Save_Variable(work_path,'All_Orien_Corr_Matrix',Corr_Matrix)
#%% 3.Sort Best Orientation and Plot all Frames.
Corr_Matrix['Best_Angle'] = Corr_Matrix.idxmax(1)
sorted_mat = Corr_Matrix.sort_values(by=['Best_Angle'])
Corr_Matrix = Corr_Matrix.drop(['Best_Angle'],axis = 1)
#%%
plt.clf()
plt.cla()
fig, ax = plt.subplots(figsize=(15,3.5),dpi = 180)
heatmap = sns.heatmap(sorted_mat.iloc[:,:-1], ax=ax,xticklabels=False,yticklabels=False,center = 0)
ax.set_xticks([0,45,90,135,180])
ax.set_xticklabels([0,45,90,135,180])
ax.set_title('All UMAP Orientation Repeats Correlation with Predicted Stim Map')
#%% 4.Find with 4 examples.
best_locs = Corr_Matrix.idxmax(1)
def Best_Series_Finder(anglemin = 10,anglemax = 15):
    satistied_series = np.where((best_locs>anglemin)*(best_locs<anglemax))[0]
    best_id = Corr_Matrix.loc[satistied_series,:].max(1).idxmax()
    origin_class = umap_predicted_label[best_id]
    origin_frame = ac.Generate_Weighted_Cell(all_orien_frame[best_id,:])
    corr_series = Corr_Matrix.loc[best_id,:]
    best_orien = corr_series.idxmax()
    best_corr = corr_series.max()
    print(f'Best Orientation {best_orien}, with corr {best_corr}.')
    print(f'UMAP Classified Class:{origin_class}')
    return origin_frame,origin_class,corr_series,best_orien,best_corr
temp1,temp1_class,temp1_corr,temp1_best_orien,temp1_best_corr = Best_Series_Finder(10,15)
temp2,temp2_class,temp2_corr,temp2_best_orien,temp2_best_corr = Best_Series_Finder(55,60)
temp3,temp3_class,temp3_corr,temp3_best_orien,temp3_best_corr = Best_Series_Finder(100,105)
temp4,temp4_class,temp4_corr,temp4_best_orien,temp4_best_corr = Best_Series_Finder(155,165)
#%% Plot 4 examples.
value_max = 6
value_min = -3
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6,6),dpi = 180)
cbar_ax = fig.add_axes([.97, .25, .03, .5])
sns.heatmap(temp1,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,0],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
axes[0,0].set_title('Example Frame 1')
sns.heatmap(temp2,center = 0,xticklabels=False,yticklabels=False,ax = axes[0,1],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
axes[0,1].set_title('Example Frame 2')
sns.heatmap(temp3,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,0],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
axes[1,0].set_title('Example Frame 3')
sns.heatmap(temp4,center = 0,xticklabels=False,yticklabels=False,ax = axes[1,1],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
axes[1,1].set_title('Example Frame 4')
fig.tight_layout()
plt.show()
#%% Plot Correlation Plots.
example_corr_frame = pd.DataFrame(columns=['Angle','Corr','Frame Name'])
for i in range(180):
    example_corr_frame.loc[len(example_corr_frame),:] = [i,temp1_corr[i],'Example 1']
    example_corr_frame.loc[len(example_corr_frame),:] = [i,temp2_corr[i],'Example 2']
    example_corr_frame.loc[len(example_corr_frame),:] = [i,temp3_corr[i],'Example 3']
    example_corr_frame.loc[len(example_corr_frame),:] = [i,temp4_corr[i],'Example 4']
plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11,3),dpi = 180)
sns.lineplot(data = example_corr_frame,x = 'Angle',y = 'Corr',hue = 'Frame Name',ax = ax)
sns.move_legend(ax,  "upper left", bbox_to_anchor=(1,0.8))
fig.tight_layout()
plt.show()