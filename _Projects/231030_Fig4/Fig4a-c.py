'''
First part of Fig4. This part will calculate average waiting time of all spontaneous events.This will be no difference between this and shuffle.

'''

#%% Initialization
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

work_path = r'D:\_Path_For_Figs\Fig4_Timecourse_Information'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Datas_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
#%% get each spon repeat frame on parameters above, and save them in fig2 folder. This is a package.
all_stim_frame_dic = ot.Load_Variable(r'D:\_Path_For_Figs\Fig2_UMAP_Pattern_Recognition','All_Stim_Frame.pkl')
all_spon_prediction = {}
all_model = {}
for i,c_loc in tqdm(enumerate(all_path_dic)):
    c_loc_name = c_loc.split('\\')[-1]
    c_reducer = ot.Load_Variable(c_loc,'All_Stim_UMAP_3D_20comp.pkl')
    all_model[c_loc_name] = c_reducer
    c_spon_frame = ot.Load_Variable(c_loc,'Spon_Before.pkl')
    c_spon_embeddings = c_reducer.transform(c_spon_frame)
    c_stim_embeddings = c_reducer.transform(all_stim_frame_dic[c_loc_name][0])
    c_stim_label = all_stim_frame_dic[c_loc_name][1]
    classifier,score = SVM_Classifier(embeddings=c_stim_embeddings,label = c_stim_label)
    predicted_spon_label = SVC_Fit(classifier,data = c_spon_embeddings,thres_prob = 0)
    all_spon_prediction[c_loc_name] = (c_spon_frame,c_spon_embeddings,predicted_spon_label)
ot.Save_Variable(r'D:\_Path_For_Figs\Fig2_UMAP_Pattern_Recognition','All_UMAP_Models',all_model)
ot.Save_Variable(r'D:\_Path_For_Figs\Fig2_UMAP_Pattern_Recognition','All_UMAP_Spon_Prediction',all_spon_prediction)
#%%################################ FigA, WHOLE NETWORK TIMES ##############################
all_locs = list(all_spon_prediction.keys())
all_spon_waittime = pd.DataFrame(columns=['From_Loc','Dist','Type'])
for i,cloc in tqdm(enumerate(all_locs)):
    cc_spon_series = all_spon_prediction[cloc][2]
    # all spon events
    all_spon_series = cc_spon_series>0
    _,start_index = All_Start_Time(all_spon_series)
    c_wait_times = Wait_Time_Distribution(start_index)
    for j,c_time in enumerate(c_wait_times):
        all_spon_waittime.loc[len(all_spon_waittime),:] = [cloc,c_time,'All']
    # all OD events
    all_od_series = (cc_spon_series>0)*(cc_spon_series<9)
    _,start_index = All_Start_Time(all_od_series)
    c_wait_times = Wait_Time_Distribution(start_index)
    for j,c_time in enumerate(c_wait_times):
        all_spon_waittime.loc[len(all_spon_waittime),:] = [cloc,c_time,'OD']
    # all Orien events
    all_orien_series = (cc_spon_series>8)*(cc_spon_series<17)
    _,start_index = All_Start_Time(all_orien_series)
    c_wait_times = Wait_Time_Distribution(start_index)
    for j,c_time in enumerate(c_wait_times):
        all_spon_waittime.loc[len(all_spon_waittime),:] = [cloc,c_time,'Orien']
# Plot weibull fit map.
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(9.5,5),dpi = 180, sharex='col')
    ## All plot
all_waittime = np.array(all_spon_waittime.groupby('Type').get_group('All')['Dist']).astype('f8')
params = stats.exponweib.fit(all_waittime,floc = 1)
x = np.linspace(0, 100, 300)
pdf_fitted = stats.exponweib.pdf(x, *params)
axes[0,0].hist(all_waittime, bins=30, density=True, alpha=1, label='Data')
# plt.plot(x, stats.gamma.pdf(x, a=shape, loc=loc, scale=scale))
axes[0,0].plot(x, pdf_fitted, 'r-', label='Fitted')
axes[0,0].set_title('All wait time distribution')
# And QQ Plot
stats.probplot(all_waittime, dist=stats.exponweib,sparams = params, plot=axes[1,0], rvalue=True)
axes[1,0].set_title('QQ plot')
axes[1,0].set_xlabel('Theoretical quantiles')
axes[1,0].set_ylabel('Sample quantiles')
    ## Orien plot
orien_waittime = np.array(all_spon_waittime.groupby('Type').get_group('Orien')['Dist']).astype('f8')
params = stats.exponweib.fit(orien_waittime,floc = 0.5)
x = np.linspace(0, 100, 300)
pdf_fitted = stats.exponweib.pdf(x, *params)
axes[0,1].hist(orien_waittime, bins=30, density=True, alpha=1, label='Data')
# plt.plot(x, stats.gamma.pdf(x, a=shape, loc=loc, scale=scale))
axes[0,1].plot(x, pdf_fitted, 'r-', label='Fitted')
axes[0,1].set_title('Orientation wait time distribution')
# And QQ Plot
stats.probplot(orien_waittime, dist=stats.exponweib,sparams = params, plot=axes[1,1], rvalue=True)
axes[1,1].set_title('QQ plot')
axes[1,1].set_xlabel('Theoretical quantiles')
axes[1,1].set_ylabel('Sample quantiles')
    ## OD plot
od_waittime = np.array(all_spon_waittime.groupby('Type').get_group('OD')['Dist']).astype('f8')
params = stats.exponweib.fit(od_waittime,floc = -1)
x = np.linspace(0, 300, 900)
pdf_fitted = stats.exponweib.pdf(x, *params)
axes[0,2].hist(od_waittime, bins=30, density=True, alpha=1, label='Data')
# plt.plot(x, stats.gamma.pdf(x, a=shape, loc=loc, scale=scale))
axes[0,2].plot(x, pdf_fitted, 'r-', label='Fitted')
axes[0,2].set_title('OD wait time distribution')
# And QQ Plot
stats.probplot(od_waittime, dist=stats.exponweib,sparams = params, plot=axes[1,2], rvalue=True)
axes[1,2].set_title('QQ plot')
axes[1,2].set_xlabel('Theoretical quantiles')
axes[1,2].set_ylabel('Sample quantiles')

fig.suptitle('Wait Time Distribution')
fig.tight_layout()
plt.show()
ot.Save_Variable(work_path,'All_Spon_Waittime',all_spon_waittime)
#%%##################### FigB, OD-ORIENTATION CSD #####################################
from scipy.signal import csd
all_freq_power = pd.DataFrame(columns=['FromLoc','Frequency(Hz)','Cross Power Spectrum'])
for i,cloc in tqdm(enumerate(all_locs)):
    cc_spon_series = all_spon_prediction[cloc][2]
    c_od_series = (cc_spon_series>0)*(cc_spon_series<9)
    c_orien_series = (cc_spon_series>8)*(cc_spon_series<17)
    frequencies, cross_power_spectrum = csd(c_od_series,c_orien_series,fs = 1.301)
    for j,c_freq in enumerate(frequencies):
        all_freq_power.loc[len(all_freq_power),:] = [cloc,c_freq,abs(cross_power_spectrum[j])]
# plot csd graphs.
plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4),dpi = 180)
csd_plot = sns.lineplot(data = all_freq_power,x = 'Frequency(Hz)',y = 'Cross Power Spectrum',ax = ax)
csd_plot.set(yscale = 'log',ylim = (0.002,0.05))
ax.set_yticks([0.05,0.01,0.005,0.002])
ax.set_yticklabels([0.05,0.01,0.005,0.002])
ax.set_title('Cross Power Specturn between OD and Orientation')

#%%####################### FigC, OD-ORIEN WAITTIME DISP ###############################
N = 1000
all_od_orien_waittime = {} # Time lag of OD with nearest Orientation.
all_orien_od_waittime = {} # Time lag of Orientation with nearest OD.
global_od_num = 0
global_orien_num = 0
for i,cloc in tqdm(enumerate(all_locs)):
    cc_spon_series = all_spon_prediction[cloc][2]
    series_len = len(cc_spon_series)
    c_od_series = (cc_spon_series>0)*(cc_spon_series<9)
    c_orien_series = (cc_spon_series>8)*(cc_spon_series<17)
    # all event waiting time.
    od_lens,all_od_locs = All_Start_Time(c_od_series)
    orien_lens,all_orien_locs = All_Start_Time(c_orien_series)
    all_od_locs = np.array(all_od_locs)
    all_orien_locs = np.array(all_orien_locs)
    od_nearest_gap = np.zeros(len(all_od_locs))
    orien_nearest_gap = np.zeros(len(all_orien_locs))
    for j in range(len(od_nearest_gap)):
        c_od_loc = all_od_locs[j]
        od_nearest_gap[j] = abs(all_orien_locs-c_od_loc).min()
    for j in range(len(orien_nearest_gap)):
        c_orien_loc = all_orien_locs[j]
        orien_nearest_gap[j] = abs(all_od_locs-c_orien_loc).min()
    # shuffle N times, and save shuffled results here.
    all_od_nearest_gap_s = np.zeros(shape = (len(all_od_locs),N))
    all_orien_nearest_gap_s = np.zeros(shape = (len(all_orien_locs),N))
    for j in range(N):
        c_od_shuffle = Random_Series_Generator(series_len,np.array(od_lens))
        c_orien_shuffle = Random_Series_Generator(series_len,np.array(orien_lens))
        _,all_od_locs_s = All_Start_Time(c_od_shuffle)
        _,all_orien_locs_s = All_Start_Time(c_orien_shuffle)
        for k in range(len(all_od_locs_s)):
            c_od_loc = all_od_locs_s[k]
            all_od_nearest_gap_s[k,j] = abs(all_orien_locs-c_od_loc).min()
        for k in range(len(all_orien_locs_s)):
            c_orien_loc = all_orien_locs_s[k]
            all_orien_nearest_gap_s[k,j] = abs(all_od_locs-c_orien_loc).min()
    # K-S test here, this result will put into result too.
    population_od = all_od_nearest_gap_s.flatten()
    population_orien = all_orien_nearest_gap_s.flatten()
    sample_od = np.array(od_nearest_gap)
    sample_orien = np.array(orien_nearest_gap)
    od_ks_result = stats.ks_2samp(sample_od,population_od)
    orien_ks_result = stats.ks_2samp(sample_orien,population_orien)
    all_od_orien_waittime[cloc] = (sample_od,population_od,od_ks_result)
    all_orien_od_waittime[cloc] = (sample_orien,population_orien,orien_ks_result)
    global_od_num += (len(sample_od)+len(population_od))
    global_orien_num += (len(sample_orien)+len(population_orien))
#%% make data above into a pd array.
global_od_waittime = pd.DataFrame(0,index = range(global_od_num),columns = ['From_Loc','Time','Type'])
global_orien_waittime = pd.DataFrame(0,index = range(global_orien_num),columns = ['From_Loc','Time','Type'])
c_start_time_od =  0
c_start_time_orien =  0

for i,cloc in enumerate(all_locs):
    #part1 od
    c_od_waittime = all_od_orien_waittime[cloc]
    real_od_num = len(c_od_waittime[0])
    shuffle_od_num = len(c_od_waittime[1])
    global_od_waittime.iloc[c_start_time_od:c_start_time_od+real_od_num,0]=cloc
    global_od_waittime.iloc[c_start_time_od:c_start_time_od+real_od_num,1] = c_od_waittime[0]
    global_od_waittime.iloc[c_start_time_od:c_start_time_od+real_od_num,2] = 'Real_Data'
    c_start_time_od += real_od_num
    global_od_waittime.iloc[c_start_time_od:c_start_time_od+shuffle_od_num,0]=cloc
    global_od_waittime.iloc[c_start_time_od:c_start_time_od+shuffle_od_num,1] = c_od_waittime[1]
    global_od_waittime.iloc[c_start_time_od:c_start_time_od+shuffle_od_num,2] = 'Shuffle_Data'
    c_start_time_od += shuffle_od_num
    # Part2 orientation
    c_orien_waittime = all_orien_od_waittime[cloc]
    real_orien_num = len(c_orien_waittime[0])
    shuffle_orien_num = len(c_orien_waittime[1])
    global_orien_waittime.iloc[c_start_time_orien:c_start_time_orien+real_orien_num,0]=cloc
    global_orien_waittime.iloc[c_start_time_orien:c_start_time_orien+real_orien_num,1] = c_orien_waittime[0]
    global_orien_waittime.iloc[c_start_time_orien:c_start_time_orien+real_orien_num,2] = 'Real_Data'
    c_start_time_orien += real_orien_num
    global_orien_waittime.iloc[c_start_time_orien:c_start_time_orien+shuffle_orien_num,0]=cloc
    global_orien_waittime.iloc[c_start_time_orien:c_start_time_orien+shuffle_orien_num,1] = c_orien_waittime[1]
    global_orien_waittime.iloc[c_start_time_orien:c_start_time_orien+shuffle_orien_num,2] = 'Shuffle_Data'
    c_start_time_orien += shuffle_orien_num
#%% Plot QQ plot.
import statsmodels.api as sm
# dist1 = np.array(global_orien_waittime.groupby('Type').get_group('Real_Data')['Time'])
# dist2 = np.array(global_orien_waittime.groupby('Type').get_group('Shuffle_Data')['Time'])
fig, ax = plt.subplots(figsize=(6, 6))
# for i,cloc in enumerate(all_locs):
    # c_location_waittime = global_orien_waittime.groupby('From_Loc').get_group(cloc)
c_location_waittime = global_od_waittime
dist1 = c_location_waittime.groupby('Type').get_group('Shuffle_Data')['Time']
dist2 = c_location_waittime.groupby('Type').get_group('Real_Data')['Time']
sorted_dist1 = np.sort(dist1)
sorted_dist2 = np.sort(dist2)
sm.qqplot_2samples(sorted_dist1, sorted_dist2, line='45',ax = ax)
ax.set_xlabel('Shuffled Waittime')
ax.set_ylabel('Real Waittime')
ax.set_title('OD-Orientation QQ Plot')
    # x = sm.ProbPlot(dist1)
    # y = sm.ProbPlot(dist2)
    # sm.qqplot_2samples(x,y, xlabel="x", ylabel="y")

    
