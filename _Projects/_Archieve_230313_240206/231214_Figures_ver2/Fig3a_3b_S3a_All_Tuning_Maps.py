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
from Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *

work_path = r'D:\_Path_For_Figs\2401_Amendments\Fig3_New'
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
#%% Plot gif of all location response in all locations.
from matplotlib.animation import FuncAnimation
all_loc_graph_path = ot.join(work_path,'All_Orientation_Maps')
ot.mkdir(all_loc_graph_path)
for i,cloc in tqdm(enumerate(all_path_dic)):
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

#%%########################FIG 3A-EXAMPLE CORR FRAMES.############################################
'''
This part will generate 4 example spon frame and their corr with all orientation maps.
'''
# 1. Generate all spon correlation matrix.
all_orien_response = ot.Load_Variable(work_path,'All_Orien_Response.pkl')
all_cell_best_oriens = ot.Load_Variable(work_path,'All_Cell_Best_Oriens')

cloc = all_path_dic[2]
cloc_name = cloc.split('\\')[-1]
ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
c_model = ot.Load_Variable(cloc,'All_Stim_UMAP_3D_20comp.pkl')
analyzer = Classify_Analyzer(ac = ac,umap_model=c_model,spon_frame=c_spon)
analyzer.Train_SVM_Classifier()
spon_label = analyzer.spon_label
#%%
c_orien_response = all_orien_response[cloc_name]
# calculate corr matrix.
Corr_Matrix = pd.DataFrame(0.0,index=range(len(c_spon)),columns=range(180))
for i in tqdm(range(len(c_spon))):
    single_spon = np.array(c_spon)[i,:]
    for j in range(180):
        c_orien_map = np.array(c_orien_response.iloc[:,j])
        # c_r,_ = stats.pearsonr(single_spon,c_orien_map)
        cos_sim = single_spon.dot(c_orien_map) / (np.linalg.norm(single_spon) * np.linalg.norm(c_orien_map))
        # squared_diff = np.square(single_spon/np.linalg.norm(single_spon) - c_orien_map/np.linalg.norm(c_orien_map))
        # squared_diff = np.square(single_spon - c_orien_map)
        # sum_squared_diff = np.sum(squared_diff)
        # euclidean_dist = np.sqrt(sum_squared_diff)
        # Corr_Matrix.loc[i,j] = c_r
        Corr_Matrix.loc[i,j] = cos_sim
        # Corr_Matrix.loc[i,j] = -euclidean_dist
#%% Normalize corr matrix by the max value.
Corr_Matrix_Norm = copy.deepcopy(Corr_Matrix)
for i in range(180):
    c_oren_disp = np.array(Corr_Matrix.iloc[:,i])
    # c_min = np.array(Corr_Matrix.iloc[:,i]).min()
    # Corr_Matrix_Norm.iloc[:,i] = Corr_Matrix_Norm.iloc[:,i]/c_max
    # Corr_Matrix_Norm.iloc[:,i] = (Corr_Matrix_Norm.iloc[:,i]-c_oren_disp.mean())/c_oren_disp.std()
    Corr_Matrix_Norm.iloc[:,i] = (Corr_Matrix_Norm.iloc[:,i])/c_oren_disp.std()
#%% 2. get 4 example locations.

def Best_Series_Finder(anglemin = 10,anglemax = 15,min_corr = 1): # calculate best corr frame in given range.
    find_from = Corr_Matrix_Norm
    find_from = find_from[find_from.min(1)<min_corr]# avoid all ensemble frames.
    best_locs = find_from.idxmax(1)
    satistied_series = np.where((best_locs>anglemin)*(best_locs<anglemax))[0]
    # best_id = Corr_Matrix_Norm.loc[satistied_series,:].max(1).idxmax()
    best_id = find_from.iloc[satistied_series,:].max(1).idxmax()
    origin_class = spon_label[best_id]
    origin_frame = ac.Generate_Weighted_Cell(c_spon.iloc[best_id,:])
    corr_series = Corr_Matrix_Norm.loc[best_id,:]
    best_orien = corr_series.idxmax()
    best_corr = corr_series.max()
    print(f'Best Orientation {best_orien}, with corr {best_corr}.')
    print(f'UMAP Classified Class:{origin_class}')
    return origin_frame,origin_class,corr_series,best_orien,best_corr
temp1,temp1_class,temp1_corr,temp1_best_orien,temp1_best_corr = Best_Series_Finder(10,15)
temp2,temp2_class,temp2_corr,temp2_best_orien,temp2_best_corr = Best_Series_Finder(55,60)
temp3,temp3_class,temp3_corr,temp3_best_orien,temp3_best_corr = Best_Series_Finder(100,105)
temp4,temp4_class,temp4_corr,temp4_best_orien,temp4_best_corr = Best_Series_Finder(155,165)
# Plot 4 examples.
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
#%% and plot corr plot of examples.
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
#%%################## FIG 3B- ALL SPON CORR WITH ORIEN #####################################
# first, seperate frame into high and low corr group.
seperate_para = 1
big_parts = Corr_Matrix_Norm[Corr_Matrix_Norm.max(1)>seperate_para]
small_parts = Corr_Matrix_Norm[Corr_Matrix_Norm.max(1)<seperate_para]

all_orien_label = np.where((spon_label>8)*(spon_label<17))[0]
all_od_label = np.where((spon_label>0)*(spon_label<9))[0]
all_color_label = np.where((spon_label>16))[0]
isi_label = np.where((spon_label==0))[0]
# orien_corrs = Corr_Matrix_Norm.iloc[all_orien_label,:]
# od_corrs = Corr_Matrix_Norm.iloc[all_od_label,:]
# color_corrs = Corr_Matrix_Norm.iloc[all_color_label,:]
# isi_corrs = Corr_Matrix_Norm.iloc[isi_label,:]

# used_corr = isi_corrs
# used_corr = orien_corrs
# used_corr = Corr_Matrix_Norm
# used_corr['Best_Angle'] = used_corr.idxmax(1)
# sorted_mat = used_corr.sort_values(by=['Best_Angle'])
big_parts['Best_Angle'] = big_parts.idxmax(1)
small_parts['Best_Angle'] = small_parts.idxmax(1)
sorted_mat_a = big_parts.sort_values(by=['Best_Angle'])
sorted_mat_b = small_parts.sort_values(by=['Best_Angle'])
sorted_mat = pd.concat([sorted_mat_a,sorted_mat_b])
# sorted_mat['Max'] = sorted_mat.max(1)
# sorted_mat = sorted_mat.sort_values(by=['Max'])
sorted_mat = sorted_mat.drop(['Best_Angle'],axis = 1)
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
#%% ################### FIG S3B - CORR WITH G16 FRAMES #########################
stim_Frames = analyzer.stim_frame
stim_labels = analyzer.stim_label
# g16_ids = np.where((stim_labels>8)*(stim_labels<17))[0]
g16_ids = np.where((stim_labels==11)+(stim_labels==13)+(stim_labels==15)+(stim_labels==9))[0]
G16_Frames = stim_Frames.loc[g16_ids,:]
G16_Corr_Matrix = pd.DataFrame(0.0,index=range(len(G16_Frames)),columns=range(180))
for i in tqdm(range(len(G16_Frames))):
    single_spon = np.array(G16_Frames)[i,:]
    for j in range(180):
        c_orien_map = np.array(c_orien_response.iloc[:,j])
        cos_sim = single_spon.dot(c_orien_map) / (np.linalg.norm(single_spon) * np.linalg.norm(c_orien_map))
        G16_Corr_Matrix.loc[i,j] = cos_sim
G16_Corr_Matrix_Norm = copy.deepcopy(G16_Corr_Matrix)
for i in range(180):
    c_oren_disp = np.array(G16_Corr_Matrix.iloc[:,i])
    G16_Corr_Matrix_Norm.iloc[:,i] = (G16_Corr_Matrix_Norm.iloc[:,i])/c_oren_disp.std()
#%%
# G16_Corr_Matrix_Norm = G16_Corr_Matrix_Norm.astype('f8')
G16_Corr_Matrix_Norm = G16_Corr_Matrix_Norm[G16_Corr_Matrix_Norm.max(1)>1]
G16_Corr_Matrix_Norm['Best_Angle'] = G16_Corr_Matrix_Norm.idxmax(1)
sorted_mat = G16_Corr_Matrix_Norm.sort_values(by=['Best_Angle'])
sorted_mat = sorted_mat.drop(['Best_Angle'],axis = 1)
G16_Corr_Matrix_Norm = G16_Corr_Matrix_Norm.drop(['Best_Angle'],axis = 1)

plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4),dpi = 180)
sns.heatmap(sorted_mat.iloc[:,:-1],center = 0,xticklabels=False,yticklabels=False,ax = ax)
ax.set_title('0-45-90-135 Stim Response vs All Orientation Maps')
ax.set_ylabel('Frames')
ax.set_xticks([0,45,90,135])
ax.set_xticklabels([0,45,90,135])
ax.set_xlabel('Orientation Angles')