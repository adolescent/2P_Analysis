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
#%% Adjust calculation of all stim graphs. only need to do this once.
# for i,c_loc in tqdm(enumerate(all_path_dic)):
#     ac = ot.Load_Variable_v2(c_loc,'Cell_Class.pkl')
#     ac.Calculate_All_T_Graphs()
#     ac.Calculate_Cell_Tunings()
#     ot.Save_Variable(c_loc,'Cell_Class',ac)
#%%######################### BASIC FUNCTIONS ###################################
def dFF(F_series,method = 'least',prop=0.1): # dFF method can be changed here.
    if method == 'least':
        base_num = int(len(F_series)*prop)
        base_id = np.argpartition(F_series, base_num)[:base_num]
        base = F_series[base_id].mean()
    dff_series = (F_series-base)/base
    return dff_series,base

def Generate_F_Series(ac,runname = '1-001',start_time = 0,stop_time = 999999,filter_para = (0.05*2/1.301,0.3*2/1.301)):
    acd = ac.all_cell_dic
    acn = ac.acn
    
    stop_time = min(len(acd[1][runname]),stop_time)
    # get all F frames first
    F_frames_all = np.zeros(shape = (stop_time-start_time,len(acn)),dtype='f8')
    for j,cc in enumerate(acn):
        c_series_raw = acd[cc][runname][start_time:stop_time]
        c_series_all = Signal_Filter(c_series_raw,order=7,filter_para=filter_para)
        F_frames_all[:,j] = c_series_all
    # then cut ON parts if needed.
    output_series = F_frames_all
    return output_series

def dFF_Matrix(F_matrix,method = 'least',prop=0.1):
    dFF_Matrix = np.zeros(shape = F_matrix.shape,dtype = 'f8')
    for i in range(F_matrix.shape[1]):
        c_F_series = F_matrix[:,i]
        c_dff_series,_ = dFF(c_F_series,method,prop)
        dFF_Matrix[:,i] = c_dff_series
    return dFF_Matrix

def dff_Matrix_Select(dff_Matrix,ac,runname='1-006',part = 'ON'):
    sfa = ac.Stim_Frame_Align
    c_stim = sfa['Run'+runname[2:]]
    c_stim = np.array(c_stim['Original_Stim_Train'])
    if part == 'ON':
        cutted_series = dff_Matrix[np.where(c_stim != -1)[0]]
    elif part == 'OFF':
        cutted_series = dff_Matrix[np.where(c_stim == -1)[0]]
    return cutted_series

def Avr_Best_Prop(dff_Matrix,best_prop = 0.1):
    dFF_max = np.zeros(dff_Matrix.shape[1],dtype = 'f8')
    for i in range(dff_Matrix.shape[1]):
        c_dff = dff_Matrix[:,i]
        c_dff_sorted = np.sort(c_dff)[::-1]
        top_percent = c_dff_sorted[:int(len(c_dff_sorted)*best_prop)]
        dFF_max[i] = top_percent.mean()
    return dFF_max
#%%######################## 1. STATS OF STIM RATIO#################################
'''
This will do stats on all cells' stim and spon response. Let's see the prop of all stim cell and spon cell.
'''
ac_spon_stim_strength = pd.DataFrame(columns = ['Loc','Cell','Spon','OD','Orien','Color'])
for i,cloc in tqdm(enumerate(all_path_dic)):
    # read in data
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    # get F series
    spon_start = c_spon.index[0]
    spon_series = Generate_F_Series(ac = ac,runname='1-001',start_time=spon_start)
    od_series = Generate_F_Series(ac = ac,runname=ac.odrun)
    orien_series = Generate_F_Series(ac = ac,runname=ac.orienrun)
    color_series = Generate_F_Series(ac = ac,runname=ac.colorrun)
    # calculate dff
    spon_dff = dFF_Matrix(spon_series)
    od_dff = dFF_Matrix(od_series)
    orien_dff = dFF_Matrix(orien_series)
    color_dff = dFF_Matrix(color_series)
    # cut dff
    od_dff = dff_Matrix_Select(od_dff,ac,ac.odrun,'ON')
    orien_dff = dff_Matrix_Select(orien_dff,ac,ac.orienrun,'ON')
    color_dff = dff_Matrix_Select(color_dff,ac,ac.colorrun,'ON')
    # calculate strength 
    spon_strength = Avr_Best_Prop(spon_dff)
    od_strength = Avr_Best_Prop(od_dff)
    orien_strength = Avr_Best_Prop(orien_dff)
    color_strength = Avr_Best_Prop(color_dff)
    # save this into a dataframe.
    for j in range(len(spon_strength)):
        ac_spon_stim_strength.loc[len(ac_spon_stim_strength),:] = [cloc_name,j+1,spon_strength[j],od_strength[j],orien_strength[j],color_strength[j]]

ac_spon_stim_strength['Best_Stim'] = ac_spon_stim_strength.loc[:,['OD','Orien','Color']].max(1)
ac_spon_stim_strength['Spon_Index'] = ac_spon_stim_strength['Spon']/ac_spon_stim_strength['Best_Stim']
# plot all location spon index distribution.
plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4),dpi = 180)
ax.axvline(x = 1,color = 'gray', linestyle = '--')
sns.histplot(ac_spon_stim_strength['Spon_Index'],ax = ax)
ax.title.set_text('Spon-Stim Ratio')
#%%############################ 2.TUNING DIFF BETWEEN STIM & SPON CELL#############################
#1. Generate all cell tuning data with spon index in it.
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
        if ac_tuning_p[cc][c_od_max]>p_thres[0] or c_od_max_value<0:
            tuned_frame.loc[len(tuned_frame),:] = [cc,'Eye','No_Tuning',c_od_max_value]
        else:
            tuned_frame.loc[len(tuned_frame),:] = [cc,'Eye',c_od_max,c_od_max_value]
        c_orien_max = c_response[3:11].idxmax()
        c_orien_max_value = c_response[c_orien_max]
        if ac_tuning_p[cc][c_orien_max]>p_thres[1] or c_orien_max_value<0:
            tuned_frame.loc[len(tuned_frame),:] = [cc,'Orien','No_Tuning',c_orien_max_value]
        else:
            tuned_frame.loc[len(tuned_frame),:] = [cc,'Orien',c_orien_max,c_orien_max_value]
        c_color_max = c_response[11:17].idxmax()
        c_color_max_value = c_response[c_color_max]
        if ac_tuning_p[cc][c_color_max]>p_thres[2] or c_color_max_value<0:
            tuned_frame.loc[len(tuned_frame),:] = [cc,'Color','No_Tuning',c_color_max_value]
        else:
            tuned_frame.loc[len(tuned_frame),:] = [cc,'Color',c_color_max,c_color_max_value]
    return tuned_frame

for i,c_loc in tqdm(enumerate(all_path_dic)):
    ac = ot.Load_Variable_v2(c_loc,'Cell_Class.pkl')
    tuned_frames = Tuning_By_Type(ac,p_thres=[0.01,0.01,0.001])
    tuned_frames['Loc'] = c_loc.split('\\')[-1]
    if i == 0:
        all_tuned_frames = copy.deepcopy(tuned_frames)
    else:
        all_tuned_frames = pd.concat([all_tuned_frames, tuned_frames], ignore_index=True)
# write spon index into the lines above.
all_tuned_frames['Spon_Index'] = 0
all_tuned_frames['Spon_dff'] = 0
for i in tqdm(range(len(all_tuned_frames))):
    c_loc = all_tuned_frames.loc[i,:]['Loc']
    cc = all_tuned_frames.loc[i,:]['Cell']
    cloc_cell_response = ac_spon_stim_strength[ac_spon_stim_strength['Loc'] == c_loc]
    cc_response = cloc_cell_response[cloc_cell_response['Cell']==cc]
    all_tuned_frames.loc[i,'Spon_Index'] = cc_response['Spon_Index'].iloc[0]
    all_tuned_frames.loc[i,'Spon_dff'] = cc_response['Spon'].iloc[0]
#%%2. Plot 3 graph of all spon index.
od_tunes = all_tuned_frames.groupby('Tune_Type').get_group('Eye').reset_index(drop = True)
orien_tunes = all_tuned_frames.groupby('Tune_Type').get_group('Orien').reset_index(drop = True)
color_tunes = all_tuned_frames.groupby('Tune_Type').get_group('Color').reset_index(drop = True)

od_tunes['Spon_Cell'] = od_tunes['Spon_Index']>1
orien_tunes['Spon_Cell'] = orien_tunes['Spon_Index']>1
color_tunes['Spon_Cell'] = color_tunes['Spon_Index']>1
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7,5), gridspec_kw={'width_ratios': [1,1,1]},sharey= True)
sns.boxplot(data = od_tunes,x = 'Spon_Cell',y = 'Best_Tune_Value',ax = axes[0],width = 0.5,showfliers = True)
sns.boxplot(data = orien_tunes,x = 'Spon_Cell',y = 'Best_Tune_Value',ax = axes[1],width = 0.5,showfliers = True)
sns.boxplot(data = color_tunes,x = 'Spon_Cell',y = 'Best_Tune_Value',ax = axes[2],width = 0.5,showfliers = True)
# titles
for i in range(3):
    axes[i].set_xticklabels(['Stim Cell','Spon Cell'])
axes[0].title.set_text('OD Tuning')
axes[1].title.set_text('Orien Tuning')
axes[2].title.set_text('Color Tuning')
# p values and p locations 
_,od_p = stats.ttest_ind(list(od_tunes[od_tunes['Spon_Cell']== True]['Best_Tune_Value']),list(od_tunes[od_tunes['Spon_Cell']== False]['Best_Tune_Value']))
_,orien_p = stats.ttest_ind(list(orien_tunes[orien_tunes['Spon_Cell']== True]['Best_Tune_Value']),list(orien_tunes[orien_tunes['Spon_Cell']== False]['Best_Tune_Value']))
_,color_p = stats.ttest_ind(list(color_tunes[color_tunes['Spon_Cell']== True]['Best_Tune_Value']),list(color_tunes[color_tunes['Spon_Cell']== False]['Best_Tune_Value']))
axes[0].text(0.4,4.5, f'p = {od_p:.5f}', fontsize=10)
axes[1].text(0.4,4.5, f'p = {orien_p:.5f}', fontsize=10)
axes[2].text(0.4,4.5, f'p = {color_p:.5f}', fontsize=10)
# legends
axes[1].yaxis.set_visible(False)
axes[2].yaxis.set_visible(False)
axes[0].set_xlabel('')
axes[2].set_xlabel('')
axes[0].set_ylabel('Tuning Cohen D')
# axes[0].set_ylim(-1,5)
fig.tight_layout()
#%%#########################3.TUNING VS SPON DFF#########################################
'''
This part will confirm whether good tuned cell have lower spon response.
'''
od_tunes['Tuned'] = (od_tunes['Best_Tune'] != 'No_Tuning')
orien_tunes['Tuned'] = (orien_tunes['Best_Tune'] != 'No_Tuning')
color_tunes['Tuned'] = (color_tunes['Best_Tune'] != 'No_Tuning')


used_tune = orien_tunes
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7,5), gridspec_kw={'width_ratios': [1.5,1]},sharey= True)
sns.scatterplot(data = used_tune,x = 'Best_Tune_Value', y = 'Spon_dff',hue = 'Tuned',s = 4,ax = axes[0],hue_order=[True,False])
sns.boxplot(data = used_tune,y = 'Spon_dff',x = 'Tuned',ax = axes[1],width = 0.5,order=[True,False],showfliers = False)
fig.suptitle('Color Tuning vs Spon Index')
axes[1].yaxis.set_visible(False)

tuned_part = used_tune[used_tune['Tuned'] == True]
untuned_part = used_tune[used_tune['Tuned'] == False]
test_t,test_p = stats.ttest_ind(tuned_part['Spon_dff'],untuned_part['Spon_dff'])
axes[1].text(0.7,2, f'p = {test_p:.5f}', fontsize=10)
fig.tight_layout()
avr_dff = tuned_part['Spon_dff'].mean()-untuned_part['Spon_dff'].mean()
r,p = stats.spearmanr(used_tune['Spon_dff'],used_tune['Best_Tune_Value'])
#%%
print(f'Tuned Spon - Untuned Spon dff:{avr_dff}')
print(f'Speatman r:{r:.5f},p = {p:.5f}')