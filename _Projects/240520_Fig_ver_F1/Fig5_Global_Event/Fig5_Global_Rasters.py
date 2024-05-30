'''
This script will Plot a 2-threshold raster plot of example location
Will show the response ensemble and global spontaneous event.

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
from scipy.stats import pearsonr
import scipy.stats as stats
from scipy.signal import find_peaks,peak_widths


savepath = r'D:\_Path_For_Figs\240520_Figs_ver_F1\Fig5_Global_Event'
datapath = r'D:\_All_Spon_Data_V1'
all_path_dic = list(ot.Get_Subfolders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)

#%%
'''
Fig5A, we will generate all location's cell Raster plot. Using Raster threshold = 1
'''
thres = 1
all_spon_dics = {}
for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    c_on_frame = c_spon>thres
    all_spon_dics[cloc_name] = c_on_frame

all_loc = list(all_spon_dics.keys())

ot.Save_Variable(savepath,'All_ON_Frames',all_spon_dics)
#%% Show example ensemble.

c_ensemble = np.array(all_spon_dics[all_loc[2]].mean(1))[4700:5350]
peaks,_ = find_peaks(c_ensemble,height = 0.1,distance = 5)

label_size = 14
title_size = 18

plt.clf()
plt.cla()
fig,axes = plt.subplots(nrows=2, ncols=1,figsize = (15,7),dpi = 180,sharex= True)
sns.heatmap(np.array(all_spon_dics[all_loc[2]].T)[:,4700:5350].astype('i4'),cbar=False,ax = axes[0],cmap = 'bwr',center = 0)
axes[0].set_yticks([0,180,360,524])
axes[0].set_yticklabels([0,180,360,524],rotation = 90,fontsize = 8)
axes[0].set_ylabel(f'Cells',size = label_size)


axes[1].plot(c_ensemble)
axes[1].plot(peaks, c_ensemble[peaks], "x")
axes[1].plot(np.zeros_like(c_ensemble), "--", color="gray")


fps = 1.301
axes[1].set_xticks([0*fps,100*fps,200*fps,300*fps,400*fps,500*fps])
axes[1].set_xticklabels([0,100,200,300,400,500],fontsize = 8)
axes[1].set_ylabel(f'Event Size',size = label_size)

axes[1].set_xlabel(f'Time (s)',size = label_size)

for i in range(2):
    axes[i].yaxis.set_label_coords(-0.04, 0.5)

fig.tight_layout()
plt.show()
#%%
'''
Fig 5B, Repeat frequency and average height of peaks.
This also generate a VAR named all_peak_info, we can use it to do many calculations.
'''
peak_thres = 0.1

all_peak_info = pd.DataFrame(columns = ['Loc','Peak_Loc','Peak_Height','Peak_Width'])
for i,cloc in enumerate(all_path_dic):
    cloc_name = cloc.split('\\')[-1]
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    c_on_frame = all_spon_dics[cloc_name]
    c_ensemble = np.array(c_on_frame.mean(1))
    peaks,_ = find_peaks(c_ensemble,height = 0.1,distance = 5)
    peak_info = peak_widths(c_ensemble,peaks) # width,hald_height,left,right
    for j,c_peak in tqdm(enumerate(peaks)):
        all_peak_info.loc[len(all_peak_info),:] = [cloc_name,c_peak,c_ensemble[c_peak],peak_info[0][j]]

ot.Save_Variable(savepath,'All_ONOff_Peaks',all_peak_info)
#%% Plotable Graphs
plt.clf()
plt.cla()
fig,axes = plt.subplots(nrows=1, ncols=2,figsize = (9,5),dpi = 180)

height_mid = np.median(all_peak_info['Peak_Height'])
width_mid = np.median(all_peak_info['Peak_Width'])
axes[0].axvline(x = height_mid,linestyle = '--',color = 'gray')
axes[1].axvline(x = width_mid,linestyle = '--',color = 'gray')
axes[0].hist(all_peak_info['Peak_Height'],bins = np.linspace(0.1,1,30))
axes[1].hist(all_peak_info['Peak_Width'],bins = np.linspace(0,10,21))

axes[0].set_ylabel('Count')
axes[0].set_xlabel('Event Scale')
axes[1].set_xlabel('Peak Width (s)')


axes[1].set_xticks(np.arange(0,11,2)*1.301)
axes[1].set_xticklabels(np.arange(0,11,2))

print(f'Mid Scale:{height_mid:.3f}; Mid Width:{width_mid/1.301:.3f}s')

#%%
'''
Fig 5C, Generate Weibull fit of all waittime.

'''

all_name = list(all_spon_dics.keys())
all_waittime = []
for i,cloc in enumerate(all_name):
    c_peaks = all_peak_info[all_peak_info['Loc']==cloc]
    all_peak_time = np.array(c_peaks['Peak_Loc'])
    c_waittime = np.diff(all_peak_time)
    all_waittime.extend(c_waittime)
all_waittime = np.array(all_waittime)
#%% Plot part
def Weibul_Fit_Plotter(ax,disp,x_max):
    #fit
    params = stats.exponweib.fit(disp,floc = 0,method='mle')
    # params = stats.expon.fit(disp,floc = 0)
    # params = stats.weibull_min.fit(disp,floc = 0)
    x = np.linspace(0, x_max, 200)
    pdf_fitted = stats.exponweib.pdf(x, *params)
    # pdf_fitted = stats.expon.pdf(x, *params)
    # plot
    ax.hist(disp, bins=50, density=True, alpha=1,range=[0, x_max])
    ax.plot(x, pdf_fitted, 'r-', label='Fitted')
    ax.set_xlim(0,x_max)

    # calculate r2 at last,using QQ Plot method
    _,(slope, intercept, r) = stats.probplot(disp, dist=stats.exponweib,sparams = params,plot=None, rvalue=True)
    r2 = r**2
    return ax,params,r2

plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5),dpi = 180, sharex='col',sharey='row')
vmax = 50
c_median = np.median(all_waittime)
ax.axvline(x = c_median,color = 'gray',linestyle = '--')

ax,_,c_r2 = Weibul_Fit_Plotter(ax,all_waittime,vmax)
ax.text(vmax*0.6,0.07,f'R2 = {c_r2:.3f}')
ax.text(vmax*0.6,0.06,f'N repeat = {len(all_waittime)}')
ax.text(vmax*0.6,0.05,f'Median = {c_median/1.301:.3f} s')
ax.set_xticks(np.arange(0,50,10)*1.301)
ax.set_xticklabels(np.arange(0,50,10))
ax.set_title('Global Ensemble Waittime',size = 14)

#%%
'''
Fig S5A- Event Frequency between different cell threshold. This will show all thres and we only use Thres = 1 for most results.
'''

thres_list_cell = np.linspace(0,2,21)
thres_list_frame = np.arange(0,0.7,0.05)

freq = np.zeros(len(thres_list_cell))
all_freqs = pd.DataFrame(columns = ['Loc','Thres_Cell','Thres_Peak','Freq'])
for j,cloc in enumerate(all_path_dic):
    example_frame = c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    cloc_name = cloc.split('\\')[-1]
    for i,c_thres in enumerate(thres_list_cell):
        c_on_frame = example_frame>c_thres
        c_ensemble = np.array(c_on_frame.mean(1))
        for k,c_thres_frame in enumerate(thres_list_frame):
            peaks,_ = find_peaks(c_ensemble,height = c_thres_frame,distance = 5)
            c_freq = len(peaks)*1.301/len(c_ensemble)
            freq[i] = c_freq
            all_freqs.loc[len(all_freqs),:] = [cloc_name,c_thres,c_thres_frame,c_freq]

#
# plt.clf()
# plt.cla()
# fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (7,5),dpi = 180)
# sns.lineplot(data = all_freqs,x = 'Thres',y = 'Freq',ax = ax)
# ax.set_ylabel('Event Frequency (Hz)')
# ax.set_xlabel('Threshold (Z Score)')
# # and get repeat freq.
# target_freq = all_freqs[all_freqs['Thres']==1.0]['Freq'].mean()
# target_freq_std = all_freqs[all_freqs['Thres']==1.0]['Freq'].std()
# print(f'Threshold 1 will get {target_freq:.4f}±{target_freq_std:.4f} Hz.')

#%% Plot parts, a little fuzzy for 0.15000000002
pivot_table = all_freqs.groupby(['Thres_Cell', 'Thres_Peak'])['Freq'].mean().unstack()
plotable = np.array(pivot_table.astype('f8'))

plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (7,5),dpi = 180)
sns.heatmap(plotable,center = 0,ax = ax,vmax = 0.2)
ax.set_title('Repeat Frequency by Threshold')
ax.set_xlabel('Thres Peak')
ax.set_ylabel('Thres Cell')
ax.invert_yaxis() 
ax.set_xticks(np.arange(14)+0.5)
ax.set_xticklabels(list(np.round((np.arange(14)*0.05),2)))

ax.set_yticks(np.arange(0,22,2)+0.5)
ax.set_yticklabels(list(np.round((np.arange(0,22,2)*0.1),2)))

#%%
'''
Fig 5D, Plot relationship between Peak width and peak height.
'''

plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (7,5),dpi = 180)
all_peak_info['Peak_Height'] = all_peak_info['Peak_Height'].astype('f8')
all_peak_info['Peak_Width'] = all_peak_info['Peak_Width'].astype('f8')

# sns.regplot(data = all_peak_info,x = 'Peak_Height',y = 'Peak_Width',ax = ax,marker='o', scatter_kws={'s':3},robust = True)
sns.scatterplot(data = all_peak_info,x = 'Peak_Height',y = 'Peak_Width',ax = ax,s = 3,marker = 'o', linewidth=0,hue = 'Loc',legend = False)
# ax.legend(markerscale=3)
ax.set_ylim(0,8)
ax.set_xlim(0.05,1)
ax.set_ylabel('Peak Width',size = 12)
ax.set_xlabel('Event Scale',size = 12)

r,p = stats.pearsonr(all_peak_info['Peak_Height'],all_peak_info['Peak_Width'])
print(f'Pearson R:{r:.3f},p = {p:.5f}')

#%%
'''
Fig 5E, Plot SVM classified ratio of all global peaks.
'''
all_repeat_info = pd.DataFrame(columns = ['Loc','Thres','Ratio','Type'])

all_thres = np.linspace(0.1,0.8,31)
win_step = 0.1

for i,cloc in enumerate(all_path_dic):
    cloc_name = cloc.split('\\')[-1]
    c_peaks_all = all_peak_info[all_peak_info['Loc']==cloc_name]
    c_class = ot.Load_Variable(cloc,'All_Spon_Repeats_PCA10.pkl')
    for j,c_thres in tqdm(enumerate(all_thres)):
        c_peaks_useful = c_peaks_all[c_peaks_all['Peak_Height']>c_thres]
        c_peaks_useful = c_peaks_useful[c_peaks_useful['Peak_Height']<(c_thres+win_step)]
        peak_num_used = len(c_peaks_useful)
        c_peaks_loc = np.array(c_peaks_useful['Peak_Loc'])# this is the useful class location.

        # count all prop. of repeats.
        c_od = np.array(c_class['OD'])[c_peaks_loc.astype('i4')]>0
        c_orien = np.array(c_class['Orien'])[c_peaks_loc.astype('i4')]>0
        c_color = np.array(c_class['Color'])[c_peaks_loc.astype('i4')]>0
        c_all = c_od+c_orien+c_color

        all_repeat_info.loc[len(all_repeat_info),:] = [cloc_name,c_thres,(c_od>0).sum()/peak_num_used,'OD']
        all_repeat_info.loc[len(all_repeat_info),:] = [cloc_name,c_thres,(c_orien>0).sum()/peak_num_used,'Orien']
        all_repeat_info.loc[len(all_repeat_info),:] = [cloc_name,c_thres,(c_color>0).sum()/peak_num_used,'Color']
        all_repeat_info.loc[len(all_repeat_info),:] = [cloc_name,c_thres,(c_all>0).sum()/peak_num_used,'All']


#%% Plot results above.
plotable = all_repeat_info[all_repeat_info['Type']=='All']
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (6,5),dpi = 180)
sns.lineplot(data = plotable,x = 'Thres',y = 'Ratio',ax = ax)
ax.set_ylim(0,1)
ax.set_ylabel('Classified Ratio',size = 12)
ax.set_xlabel('Event Scale Threshold',size = 12)
