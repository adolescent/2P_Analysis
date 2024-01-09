'''

This script will compare brightness between stim and spon.
'''



#%% Import and initialization
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



all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
#%%######################### BASIC FUNCTIONS ###################################
def dFF(F_series,method = 'least',prop=0.1): # dFF method can be changed here.
    if method == 'least':
        base_num = int(len(F_series)*prop)
        base_id = np.argpartition(F_series, base_num)[:base_num]
        base = F_series[base_id].mean()
    dff_series = (F_series-base)/base
    return dff_series,base

def Generate_F_Series(ac,runname = '1-001',start_time = 0,stop_time = 999999,part = 'ON',filter_para = (0.05*2/1.301,0.3*2/1.301)):
    acd = ac.all_cell_dic
    acn = ac.acn
    sfa = ac.Stim_Frame_Align
    stop_time = min(len(acd[1][runname]),stop_time)
    # get all F frames first
    F_frames_all = np.zeros(shape = (stop_time-start_time,len(acn)),dtype='f8')
    for j,cc in enumerate(acn):
        c_series_raw = acd[cc][runname][start_time:stop_time]
        c_series_all = Signal_Filter(c_series_raw,order=7,filter_para=filter_para)
        F_frames_all[:,j] = c_series_all
    # then cut ON parts if needed.
    if part == 'All':
        output_series = F_frames_all
    else:
        c_stim = sfa['Run'+runname[2:]]
        if c_stim == None: # for real stim parts
            print('No Stim find, return Whole series.')
            output_series = F_frames_all
        else:
            c_stim = np.array(c_stim['Original_Stim_Train'])
            c_stim = c_stim[start_time:stop_time]
            if part == 'ON':
                output_series = F_frames_all[np.where(c_stim != -1)[0]]
            elif part == 'OFF':
                output_series = F_frames_all[np.where(c_stim == -1)[0]]

    return output_series

def dFF_Matrix(F_matrix,method = 'least',prop=0.1):
    dFF_Matrix = np.zeros(shape = F_matrix.shape,dtype = 'f8')
    for i in range(F_matrix.shape[1]):
        c_F_series = F_matrix[:,i]
        c_dff_series,_ = dFF(c_F_series,method,prop)
        dFF_Matrix[:,i] = c_dff_series
    return dFF_Matrix

def Avr_Best_Prop(dff_Matrix,best_prop = 0.1):
    dFF_max = np.zeros(dff_Matrix.shape[1],dtype = 'f8')
    for i in range(dff_Matrix.shape[1]):
        c_dff = dff_Matrix[:,i]
        c_dff_sorted = np.sort(c_dff)[::-1]
        top_percent = c_dff_sorted[:int(len(c_dff_sorted)*best_prop)]
        dFF_max[i] = top_percent.mean()
    return dFF_max


#%%##################### 1. TEST METHOD VELIDITY################################
# 1. get F series of 3 stim and spon.
example_path = all_path_dic[2]
ac = ot.Load_Variable_v2(example_path,'Cell_Class.pkl')

c_spon = ot.Load_Variable(example_path,'Spon_Before.pkl')
spon_start = c_spon.index[0]
spon_series = Generate_F_Series(ac = ac,runname='1-001',start_time=spon_start,part='All')[:5500]
od_series = Generate_F_Series(ac = ac,runname=ac.odrun,part='ON')
orien_series = Generate_F_Series(ac = ac,runname=ac.orienrun,part='ON')
color_series = Generate_F_Series(ac = ac,runname=ac.colorrun,part='ON')
# 2. Generate dff series of each situation.
spon_dff = dFF_Matrix(spon_series)
od_dff = dFF_Matrix(od_series)
orien_dff = dFF_Matrix(orien_series)
color_dff = dFF_Matrix(color_series)
spon_strength = Avr_Best_Prop(spon_dff)
od_strength = Avr_Best_Prop(od_dff)
orien_strength = Avr_Best_Prop(orien_dff)
color_strength = Avr_Best_Prop(color_dff)
all_cell_strength = pd.DataFrame(0,columns = ['Spon','OD','Orien','Color','Best_Stim'],index = range(len(spon_strength)))
all_cell_strength.loc[:,'Spon'] = spon_strength
all_cell_strength.loc[:,'OD'] = od_strength
all_cell_strength.loc[:,'Orien'] = orien_strength
all_cell_strength.loc[:,'Color'] = color_strength
all_cell_strength.loc[:,'Best_Stim'] = all_cell_strength.loc[:,['OD','Orien','Color']].max(1)
# 3. Plot Scatter of all parts.
plt.clf()
plt.cla()
fig = plt.figure(figsize = (15,15))
g = sns.pairplot(all_cell_strength)
g.set(xlim=(0,5), ylim = (0,5))
g.fig.suptitle('Stim-Spon Strength Compare', y=1.02,size = 24)
#%% Get stim-spon ratio, to see whether it is stable.
spon_series_after = Generate_F_Series(ac = ac,runname='1-003',start_time=spon_start,part='All')
spon_dff_after = dFF_Matrix(spon_series_after)
spon_strength_after = Avr_Best_Prop(spon_dff_after)
spon_p1 = spon_dff[-4000:-2000,:]
spon_p2 = spon_dff[-2000:,:]
spon_strength_p1 = Avr_Best_Prop(spon_p1)
spon_strength_p2 = Avr_Best_Prop(spon_p2)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9,4),dpi = 180)
for i in range(2):
    axes[i].set_xlim(0,2)
    axes[i].set_ylim(0,2)
    axes[i].plot([0,2],[0,2],color = 'gray', linestyle = '--')

axes[0].scatter(spon_strength_p1,spon_strength_p2,s = 3)
axes[0].title.set_text('Dichotomy Verify')
axes[0].set_xlabel('Early Half')
axes[0].set_ylabel('Late Half')
axes[1].scatter(spon_strength,spon_strength_after,s = 3)
axes[1].title.set_text('Before-After Verify')
axes[1].set_xlabel('Spon Before')
axes[1].set_ylabel('Spon After')
dicho_r,_ = stats.spearmanr(spon_strength_p1,spon_strength_p2)
before_after_r,_ = stats.spearmanr(spon_strength,spon_strength_after)

axes[0].text(1.1,0.3, f'Spearmanr:{dicho_r:.3f}', fontsize=10)
axes[1].text(1.1,0.3, f'Spearmanr:{before_after_r:.3f}', fontsize=10)

#%%###########################2. STIM-SPON-PROPTION#######################################
'''
This part will calculate a spon ratio of stim, and discuss tuning-ratio relationship.
'''
# 1. Plot Stim-Spon Ratio directly
all_cell_strength['Spon_Ratio'] = all_cell_strength['Spon']/all_cell_strength['Best_Stim']
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4),dpi = 180)
ax.axvline(x = 1,color = 'gray', linestyle = '--')
sns.histplot(all_cell_strength['Spon_Ratio'],ax = ax)
ax.title.set_text('Spon-Stim Ratio')
#%% 2.let's see whether this para is related to tuning strength.
# all_red_t = ac.OD_t_graphs['OD'].loc['t_value']
# all_red_p = ac.OD_t_graphs['OD'].loc['p_value']
all_red_t = ac.Color_t_graphs['Red-White'].loc['t_value']
all_red_p = ac.Color_t_graphs['Red-White'].loc['p_value']
color_tune_compare = pd.DataFrame(columns = ['Spon_Index','Tuning_Index','Tuning'])
# plt.scatter(all_red_t,all_cell_strength['Spon_Ratio'],s = 3)
thres = 0.0001
for i in range(len(all_red_t)):
    if all_red_p.iloc[i]<thres and all_red_t.iloc[i]>0:
    # if all_red_p.iloc[i]<thres:
        c_tun = 'Tuned'
    else:
        c_tun = 'Non-Tuned'
    color_tune_compare.loc[len(color_tune_compare)] = [all_cell_strength['Spon_Ratio'].iloc[i],all_red_t.iloc[i],c_tun]
    # color_tune_compare.loc[len(color_tune_compare)] = [all_cell_strength['Spon'].iloc[i],all_red_t.iloc[i],c_tun] 
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7,5), gridspec_kw={'width_ratios': [1.5,1]},sharey= True)
sns.scatterplot(data = color_tune_compare,x = 'Tuning_Index', y = 'Spon_Index',hue = 'Tuning',s = 6,ax = axes[0],hue_order=['Non-Tuned','Tuned'])
sns.boxplot(data = color_tune_compare,y = 'Spon_Index',x = 'Tuning',ax = axes[1],width = 0.5,order=['Non-Tuned','Tuned'],showfliers = True)
fig.suptitle('Color Dff vs Spon Index')
axes[1].yaxis.set_visible(False)
fig.tight_layout()

tuned_part = color_tune_compare[color_tune_compare['Tuning'] == 'Tuned']
untuned_part = color_tune_compare[color_tune_compare['Tuning'] == 'Non-Tuned']
test_t,test_p = stats.ttest_ind(tuned_part['Spon_Index'],untuned_part['Spon_Index'])
axes[1].text(0.7,2, f'p = {test_p:.5f}', fontsize=10)

#%% 3. Another vision, let's see the difference between spon cell and stim cell.
def Tuning_By_Type(ac,p_thres = [0.01,0.01,0.001]):
    print('This function can use only on full tuning points!')
    ac_tuning = ac.all_cell_tunings
    ac_tuning_p = ac.all_cell_tunings_p_value
    tuned_frame = pd.DataFrame(columns = ['Cell','Tune_Type','Best_Tune','Best_Tune_Value'])
    # od first.
    for i,cc in enumerate(ac.acn):
        c_response = ac_tuning[cc]
        c_p = ac_tuning_p[cc]
        c_od_max = c_response[:2].idxmax()
        c_od_max_value = c_response[c_od_max]
        if ac_tuning_p[cc][c_od_max]>p_thres[0]:
            tuned_frame.loc[len(tuned_frame),:] = [cc,'Eye','No_Tuning',c_od_max_value]
        else:
            tuned_frame.loc[len(tuned_frame),:] = [cc,'Eye',c_od_max,c_od_max_value]
            
        c_orien_max = c_response[3:11].idxmax()
        c_orien_max_value = c_response[c_orien_max]
        tuned_frame.loc[len(tuned_frame),:] = [cc,'Orien',c_orien_max,c_orien_max_value]
        c_color_max = c_response[11:17].idxmax()
        c_color_max_value = c_response[c_color_max]
        tuned_frame.loc[len(tuned_frame),:] = [cc,'Color',c_color_max,c_color_max_value]
    return tuned_frame

