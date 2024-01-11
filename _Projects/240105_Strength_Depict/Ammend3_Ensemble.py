'''
This will calculate ensemble of spontaneous series.
Test how to define ensemble.
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

#%%######################1.ENSEMBLE DEFINE###################
'''
This part will disscuss how to get ensemble of spon series.
'''
from scipy.signal import find_peaks,peak_widths
def Peak_Find(input_series,height_lim,dist_lim):
    # find peak locations
    peak_locs, _ = find_peaks(input_series, height=height_lim)
    # wash peaks with dist below lim
    indices_to_keep = np.where(np.diff(peak_locs) >= dist_lim)[0]
    peak_locs = peak_locs[np.concatenate(([0], indices_to_keep + 1))]
    # get hald width and heights.
    peak_heights = input_series[peak_locs]
    results_half = peak_widths(input_series,peak_locs, rel_height=0.5)
    half_widths = results_half[0]
    return peak_locs,half_widths,peak_heights,results_half

z_thres = 1
peak_height = 0.05
peak_dist = 5
all_peaks = pd.DataFrame(columns = ['Loc','Peak_id','Peak_Width','Peak_Height','Peak_Pattern'])
all_peaks_s = pd.DataFrame(columns = ['Loc','Peak_id','Peak_Width','Peak_Height','Peak_Pattern'])
all_peak_freq = []
for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    spon_start = c_spon.index[0]
    spon_series = Generate_F_Series(ac = ac,runname='1-001',start_time=spon_start)
    spon_dff = dFF_Matrix(spon_series)
    # choose z score>1 as response on.
    ensemble_mat = np.array(c_spon>z_thres)
    ensemble_curve = ensemble_mat.mean(1)
    c_peaks,c_halfwidths,c_heights,c_results_half = Peak_Find(ensemble_curve,peak_height,peak_dist)
    all_peak_freq.append(len(c_peaks)/len(c_spon))
    ## If you want to plot an example.
    # and shuffled frame here.
    c_spon_s = Spon_Shuffler(c_spon,method='phase')
    ensemble_mat_s = np.array(c_spon_s>z_thres)
    ensemble_curve_s = ensemble_mat_s.mean(1)
    c_peaks_s,c_halfwidths_s,c_heights_s,c_results_half_s = Peak_Find(ensemble_curve_s,peak_height,peak_dist)
    #x = ensemble_curve
    # plt.plot(x)
    # plt.plot(c_peaks, x[c_peaks], "x")
    # plt.hlines(*c_results_half[1:], color="gray",linestyles='--')
    # plt.show()
    for j in tqdm(range(len(c_peaks))):
        c_peak_pattern = ensemble_mat[c_peaks[j],:]
        all_peaks.loc[len(all_peaks),:] = [cloc_name,c_peaks[j],c_halfwidths[j],c_heights[j],c_peak_pattern]
        # and save shuffled too
    for j in tqdm(range(len(c_peaks_s))):
        c_peak_pattern_s = ensemble_mat_s[c_peaks_s[j],:]
        all_peaks_s.loc[len(all_peaks_s),:] = [cloc_name,c_peaks_s[j],c_halfwidths_s[j],c_heights_s[j],c_peak_pattern_s]
        

#%%2. Plot all ensemble frequency and ensemble size 
# mean peak width
all_peaks['Data'] = 'Real_Data'
all_peaks_s['Data'] = 'Shuffle'
combined_frame = pd.concat([all_peaks,all_peaks_s], ignore_index=True)


plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4),dpi = 180)
mean_width = all_peaks['Peak_Width'].mean()
ax.axvline(x = mean_width,color = 'black', linestyle = '--')
mean_width_s = all_peaks_s['Peak_Width'].mean()
ax.axvline(x = mean_width_s,color = 'black', linestyle = '--')
sns.histplot(data = combined_frame,x = 'Peak_Width',hue = 'Data',hue_order=['Real_Data','Shuffle'],ax = ax)
ax.title.set_text('Half Peak Width')
ax.text(4,350, f'Mean = {mean_width:.3f}', fontsize=10)
ax.text(4,400, f'Mean Shuffle = {mean_width_s:.3f}', fontsize=10)

# and Ensemble size
plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4),dpi = 180)
mean_height = all_peaks['Peak_Height'].mean()
ax.axvline(x = mean_height,color = 'black', linestyle = '--')
mean_height_s = all_peaks_s['Peak_Height'].mean()
ax.axvline(x = mean_height_s,color = 'gray', linestyle = '--')
sns.histplot(data = combined_frame,x = 'Peak_Height',hue = 'Data',ax = ax)
ax.title.set_text('Ensemble Size')
ax.text(0.5,500, f'Mean = {mean_height:.3f}', fontsize=10)
ax.text(0.5,700, f'Mean Shuffle = {mean_height_s:.3f}', fontsize=10)

# and scatter between height and width.

plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4),dpi = 180)
# sns.scatterplot(data = all_peaks,x = 'Peak_Height',y = 'Peak_Width',ax = ax,s = 3)
sns.scatterplot(data = combined_frame,x = 'Peak_Height',y = 'Peak_Width',hue = 'Data',ax = ax,s = 3)
c_r,c_p = stats.pearsonr(all_peaks['Peak_Height'],all_peaks['Peak_Width'])
c_r_s,c_p_s = stats.pearsonr(all_peaks_s['Peak_Height'],all_peaks_s['Peak_Width'])
ax.title.set_text('Ensemble Size vs Duration')
ax.text(0.55,9, f'Pearson R Real = {c_r:.3f}', fontsize=10)
ax.text(0.55,8, f'Pearson R Shuffle = {c_r_s:.3f}', fontsize=10)
ax.set_ylim(0,10)
#%%############################2.ENSEMBLE VS TUNING###############################
'''
This part will try to find the relationship between ensemble and tuning.
'''
# 1. data generation
# peak_groups = all_peaks.groupby('Loc')
peak_groups = combined_frame.groupby('Loc')
loc_infos = pd.DataFrame(columns = ['Loc','All','LE','RE','Orien0','Orien45','Orien90','Orien135','Red','Green','Blue'])# Number of each tuning. used for calculation.
peak_Network_Num = pd.DataFrame(columns = ['Loc','Data','All','LE','RE','Orien0','Orien45','Orien90','Orien135','Red','Green','Blue'])# In each ensemble, the strength of each network

od_thres = 0.01
orien_thres = 0.01
color_thres = 0.001

for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_peaks = peak_groups.get_group(cloc_name)
    ac_tuning = ac.all_cell_tunings
    ac_tuning_p = ac.all_cell_tunings_p_value
    ## DECREPTED.
    # LE_cells = np.array(ac_tuning_p.loc['OD']<od_thres)
    # RE_cells = np.array(ac_tuning_p.loc['OD']<od_thres)
    # Orien0_cells = np.array(ac_tuning_p.loc['Orien0-0']<orien_thres)
    # Orien45_cells = np.array(ac_tuning_p.loc['Orien45-0']<orien_thres)
    # Orien90_cells = np.array(ac_tuning_p.loc['Orien90-0']<orien_thres)
    # Orien135_cells = np.array(ac_tuning_p.loc['Orien135-0']<orien_thres)
    # red_cells = np.array(ac_tuning_p.loc['Red-White']<color_thres)
    # green_cells = np.array(ac_tuning_p.loc['Green-White']<color_thres)
    # blue_cells = np.array(ac_tuning_p.loc['Blue-White']<color_thres)
    ## use best eye and best orien as orien tools.
    LE_cells = np.array((ac_tuning.loc['OD']>0)*(ac_tuning_p.loc['OD']<od_thres))
    RE_cells = np.array((ac_tuning.loc['OD']<0)*(ac_tuning_p.loc['OD']<od_thres))
    orien_parts = ac_tuning.loc[['Orien0-0','Orien45-0','Orien90-0','Orien135-0']]
    Orien0_cells = np.array((orien_parts.idxmax() =='Orien0-0')*(ac_tuning_p.loc['Orien0-0']<orien_thres))
    Orien45_cells = np.array((orien_parts.idxmax() =='Orien45-0')*(ac_tuning_p.loc['Orien45-0']<orien_thres))
    Orien90_cells = np.array((orien_parts.idxmax() =='Orien90-0')*(ac_tuning_p.loc['Orien90-0']<orien_thres))
    Orien135_cells = np.array((orien_parts.idxmax() =='Orien135-0')*(ac_tuning_p.loc['Orien135-0']<orien_thres))
    color_parts = ac_tuning.loc[['Red-White','Green-White','Blue-White']]
    red_cells = np.array((color_parts.idxmax() =='Red-White')*(ac_tuning_p.loc['Red-White']<color_thres))
    green_cells = np.array((color_parts.idxmax() =='Green-White')*(ac_tuning_p.loc['Green-White']<color_thres))
    blue_cells = np.array((color_parts.idxmax() =='Blue-White')*(ac_tuning_p.loc['Blue-White']<color_thres))

    loc_infos.loc[len(loc_infos),:] = [cloc_name,ac_tuning.shape[1],LE_cells.sum(),RE_cells.sum(),Orien0_cells.sum(),Orien45_cells.sum(),Orien90_cells.sum(),Orien135_cells.sum(),red_cells.sum(),green_cells.sum(),blue_cells.sum()]
    # cycle all peaks in current location, and get each network activation.
    for j in range(len(c_peaks)):
        c_pattern = c_peaks.iloc[j,-2]
        data_type = c_peaks.iloc[j,-1]
        c_LE = c_pattern*LE_cells
        c_RE = c_pattern*RE_cells
        c_orien0 = c_pattern*Orien0_cells
        c_orien45 = c_pattern*Orien45_cells
        c_orien90 = c_pattern*Orien90_cells
        c_orien135 = c_pattern*Orien135_cells
        c_red = c_pattern*red_cells
        c_green = c_pattern*green_cells
        c_blue = c_pattern*blue_cells
        peak_Network_Num.loc[len(peak_Network_Num),:] = [cloc_name,data_type,c_pattern.sum(),c_LE.sum(),c_RE.sum(),c_orien0.sum(),c_orien45.sum(),c_orien90.sum(),c_orien135.sum(),c_red.sum(),c_green.sum(),c_blue.sum()]
#%% 2. calculate tuning_purity
'''
This part will generate tuning score of shuffle and real data, let's see the ensemble of .

'''
#
tuning_index_frame = pd.DataFrame(index = range(len(peak_Network_Num)*9),columns = ['Loc','Data Type','Ensemble Size','Network_Type','Network','Index'])
tuning_index_frame_best = pd.DataFrame(columns = ['Loc','Data Type','Ensemble Size','Network_Type','Network','Best Index'])
counter = 0
for i in tqdm(range(len(peak_Network_Num))):
    c_peak = peak_Network_Num.loc[i,:]
    cloc = c_peak['Loc']
    cloc_info = loc_infos[loc_infos['Loc']==cloc].iloc[0]
    c_names = ['LE','RE','Orien0','Orien45','Orien90','Orien135','Red','Green','Blue']
    c_best_index = 0
    c_best_tuning = 'None'
    c_best_tuningtype = 'None'
    for j,cc_name in enumerate(c_names):
        if j<2:
            c_type = 'Eye'
        elif j<6:
            c_type = 'Orientation'
        else:
            c_type = 'Color'
        tuning_index_frame.loc[counter,:] = [cloc,c_peak['Data'],c_peak['All']/cloc_info['All'],c_type,cc_name,(c_peak[cc_name]/c_peak['All'])/(cloc_info[cc_name]/cloc_info['All'])]
        counter +=1
        if c_best_index<(c_peak[cc_name]/c_peak['All'])/(cloc_info[cc_name]/cloc_info['All']):
            c_best_index = (c_peak[cc_name]/c_peak['All'])/(cloc_info[cc_name]/cloc_info['All'])
            c_best_tuning = cc_name
            c_best_tuningtype = c_type
    tuning_index_frame_best.loc[len(tuning_index_frame_best),:] = [cloc,c_peak['Data'],c_peak['All']/cloc_info['All'],c_best_tuningtype,c_best_tuning,c_best_index]
#%% 3. plot tuning purity with relation of ensemble size.
# plotable_data = tuning_index_frame.groupby('Loc').get_group(all_path_dic[7].split('\\')[-1])
plotable_data = tuning_index_frame[tuning_index_frame['Network_Type'] != 'Color']
# plotable_data = plotable_data[plotable_data['Network']=='Orien0']
# plotable_data = tuning_index_frame.groupby('Network_Type').get_group('Color')

plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4),dpi = 180)
# sns.histplot(data = plotable_data,x = 'Index',hue = 'Data Type',ax = ax)
# ax.set_xlim(0,5)
ax.axhline(y=1,color = 'gray', linestyle = '--')
# sns.scatterplot(data = plotable_data,x = 'Ensemble Size',y = 'Index',hue = 'Data Type',s = 3,ax = ax)
sns.histplot(data = plotable_data,y = 'Index',x = 'Ensemble Size',hue = 'Data Type',ax = ax)
# ax.set_ylim(-1,3)
#%% 4. Another method, we just calculate each network score, and try to get score relation.
peak_score_relation = pd.DataFrame(index = range(len(peak_Network_Num)),columns = ['Loc','Ensemble_Size','Data Type','LE','RE','Orien0','Orien45','Orien90','Orien135','Red','Green','Blue'])
counter = 0
for i in tqdm(range(len(peak_Network_Num))):
    c_peak = peak_Network_Num.loc[i,:]
    cloc = c_peak['Loc']
    cloc_info = loc_infos[loc_infos['Loc']==cloc].iloc[0]
    c_names = ['LE','RE','Orien0','Orien45','Orien90','Orien135','Red','Green','Blue']
    peak_score_relation.loc[counter,'Loc'] = cloc
    peak_score_relation.loc[counter,'Ensemble_Size'] = c_peak['All']/cloc_info['All']
    peak_score_relation.loc[counter,'Data Type'] = c_peak['Data']
    for j,cc_name in enumerate(c_names):
        peak_score_relation.loc[counter,cc_name] = (c_peak[cc_name]/cloc_info[cc_name])
    counter +=1
#%% 5.Plot just score method.
# plotable_data = peak_score_relation[peak_score_relation['Data Type']=='Real_Data']
# plotable_data = plotable_data.sort_values(by=['Ensemble_Size','LE','RE'],ascending=False)
plotable_data = peak_score_relation
# plotable_data['Best_Tune_Ratio'] = 0
network_a = [plotable_data['LE'],plotable_data['Orien0'],plotable_data['Orien45'],plotable_data['Red']]
network_b = [plotable_data['RE'],plotable_data['Orien90'],plotable_data['Orien135'],plotable_data['Blue']]
network_name = ['OD_ratio','HV_ratio','AO_ratio','RB_ratio']
for i,cname in enumerate(network_name):
    plotable_data[cname] = (network_a[i]-network_b[i])/(network_a[i]+network_b[i])

best_ratio_locs = abs(plotable_data[['OD_ratio','HV_ratio','AO_ratio','RB_ratio']]).idxmax(1)
for i in range(len(best_ratio_locs)):
    plotable_data['Best_Tune_Ratio'] = plotable_data[best_ratio_locs[i]]
#
plt.clf()
plt.cla()
# plotable_data['HV_ratio'] = plotable_data['HV_ratio'].astype('f8')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4),dpi = 180)
# sns.histplot(data = plotable_data,x = 'LE',y = 'RE',hue = 'Data Type',ax = ax)
ax.axhline(y=0,color = 'gray', linestyle = '--')
sns.histplot(data = plotable_data,y = 'Best_Tune_Ratio',x = 'Ensemble_Size',hue = 'Data Type',ax = ax)
# sns.histplot(data =  plotable_data,hue= 'Data Type',x = 'Best_Tune_Ratio')
# sns.heatmap((plotable_data.iloc[:,[1,3,4,5,6,7,8,9,10,11]].astype('f8')),ax = ax,center=0)
# ax.set_xlim(-2,2)
ax.title.set_text('Best Tuning Index vs Ensemble Size')


