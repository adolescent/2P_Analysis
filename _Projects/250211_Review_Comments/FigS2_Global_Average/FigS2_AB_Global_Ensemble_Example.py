'''
Calculating ensemble freqency of global ensemble and single cell.

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
from Review_Fix_Funcs import *


savepath = r'G:\我的云端硬盘\#Figs\#250211_Revision1\FigS2'
datapath = r'D:\_DataTemp\_Fig_Datas\_All_Spon_Data_V1'
all_path_dic = list(ot.Get_Subfolders(datapath))
all_path_dic.pop(4)
all_path_dic.pop(6)
#%% Step1, getting thresholded response matrix.
'''
Part 1, calculate global ensemble frequency. Stats for all location. This value is discribed in article.
'''
#%% Single Threshold ensemble estimation.
thres = 1
all_spon_dics = {}
for i,cloc in tqdm(enumerate(all_path_dic)):
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    cloc_name = cloc.split('\\')[-1]
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    start = c_spon.index[0]
    end = c_spon.index[-1]
    c_spon = Z_refilter(ac,'1-001',start,end).T
    c_on_frame = c_spon>thres
    all_spon_dics[cloc_name] = c_on_frame

all_loc = list(all_spon_dics.keys())



#%% Get Threshold with ensemble.
thres_list = np.linspace(0,2,21)
freq = np.zeros(len(thres_list))
all_freqs = pd.DataFrame(columns = ['Loc','Thres','Freq'])
for j,cloc in enumerate(all_path_dic):
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    example_frame = ot.Load_Variable(cloc,'Spon_Before.pkl')
    start = example_frame.index[0]
    end = example_frame.index[-1]
    example_frame = Z_refilter(ac,'1-001',start,end).T

    cloc_name = cloc.split('\\')[-1]
    for i,c_thres in enumerate(thres_list):
        c_on_frame = example_frame>c_thres
        c_ensemble = np.array(c_on_frame.mean(1))
        peaks,_ = find_peaks(c_ensemble,height = 0.1,distance = 5)
        c_freq = len(peaks)*1.301/len(c_ensemble)
        freq[i] = c_freq
        all_freqs.loc[len(all_freqs),:] = [cloc_name,c_thres,c_freq]

plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (7,5),dpi = 180)
sns.lineplot(data = all_freqs,x = 'Thres',y = 'Freq',ax = ax)
ax.set_ylabel('Event Frequency (Hz)')
ax.set_xlabel('Threshold (Z Score)')
# and get repeat freq.
target_freq = all_freqs[all_freqs['Thres']==1.0]['Freq'].mean()
target_freq_std = all_freqs[all_freqs['Thres']==1.0]['Freq'].std()
print(f'Threshold 1 will get {target_freq:.4f}±{target_freq_std:.4f} Hz.')

#%% Show example ensemble.

c_ensemble = np.array(all_spon_dics[all_loc[2]].mean(1))[4700:5350]
peaks,_ = find_peaks(c_ensemble,height = 0.1,distance = 5)

label_size = 14
# title_size = 18

plt.clf()
plt.cla()
fig,axes = plt.subplots(nrows=2, ncols=1,figsize = (10,7),dpi = 180,sharex= True)
sns.heatmap(np.array(all_spon_dics[all_loc[2]].T)[:,4700:5350],cbar=False,ax = axes[0],cmap='bwr',center=0,yticklabels=False)
# axes[0].set_yticks([0,180,360,524])
# axes[0].set_yticklabels([0,180,360,524],rotation = 90,fontsize = 8)
# axes[0].set_ylabel(f'Cells',size = label_size)


axes[1].plot(c_ensemble)
axes[1].plot(peaks, c_ensemble[peaks], "x")
axes[1].plot(np.zeros_like(c_ensemble), "--", color="gray")


fps = 1.301
axes[1].set_yticks([0,0.2,0.4,0.6,0.8,1])
axes[1].set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize = label_size)

axes[1].set_xticks([0*fps,100*fps,200*fps,300*fps,400*fps,500*fps])
axes[1].set_xticklabels([0,100,200,300,400,500],fontsize = label_size)
# axes[1].set_ylabel(f'Event Size',size = label_size)

# axes[1].set_xlabel(f'Time (s)',size = label_size)

for i in range(2):
    axes[i].yaxis.set_label_coords(-0.04, 0.5)

fig.tight_layout()
fig.savefig(ot.join(savepath,'FigS2AB_Global_Thresed'),bbox_inches='tight')
plt.show()
#%%######################### P2 Get ALL Peak Info ##############
thres = 1
all_on_frame = {}
all_peak_info = pd.DataFrame(columns = ['Loc','Peak_Loc','Peak_Height','Peak_Width'])

for i,cloc in enumerate(all_path_dic):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    example_frame = ot.Load_Variable(cloc,'Spon_Before.pkl')
    start = example_frame.index[0]
    end = example_frame.index[-1]
    example_frame = Z_refilter(ac,'1-001',start,end).T

    c_on_frame = np.array(example_frame>thres)
    all_on_frame[cloc_name] = c_on_frame
    c_ensemble = c_on_frame.mean(1)
    peaks,_ = find_peaks(c_ensemble,height = 0.1,distance = 5)
    peak_info = peak_widths(c_ensemble,peaks) # width,hald_height,left,right
    for j,c_peak in tqdm(enumerate(peaks)):
        all_peak_info.loc[len(all_peak_info),:] = [cloc_name,c_peak,c_ensemble[c_peak],peak_info[0][j]]

ot.Save_Variable(savepath,'All_ON_Frames',all_on_frame)
ot.Save_Variable(savepath,'All_OnOff_Peaks',all_peak_info)
# Plot part
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (7,5),dpi = 180)
all_peak_info['Peak_Height'] = all_peak_info['Peak_Height'].astype('f8')
all_peak_info['Peak_Width'] = all_peak_info['Peak_Width'].astype('f8')

# sns.regplot(data = all_peak_info,x = 'Peak_Height',y = 'Peak_Width',ax = ax,marker='o', scatter_kws={'s':3},robust = True)
sns.scatterplot(data = all_peak_info,x = 'Peak_Height',y = 'Peak_Width',ax = ax,s = 3,marker = 'o', linewidth=0)
# ax.legend(markerscale=3)
ax.set_ylim(0,8)
ax.set_xlim(0.05,1)
ax.set_ylabel('Peak Width',size = 12)
ax.set_xlabel('Event Scale',size = 12)

r,p = stats.pearsonr(all_peak_info['Peak_Height'],all_peak_info['Peak_Width'])
print(f'Pearson R:{r:.3f},p = {p:.5f}')




#%% and plot height and width distribution.
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

#%% Wait time of global ensemble.
all_name = list(all_on_frame.keys())
all_waittime = []
for i,cloc in enumerate(all_name):
    c_peaks = all_peak_info[all_peak_info['Loc']==cloc]
    all_peak_time = np.array(c_peaks['Peak_Loc'])
    c_waittime = np.diff(all_peak_time)
    all_waittime.extend(c_waittime)
all_waittime = np.array(all_waittime)

#%% fit weibul
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
fig.savefig(ot.join(savepath,'FigS2D_Global_Waittime.png'),bbox_inches='tight')
#%%
########## Plot jointplot of all peaks' height and width


        
plt.clf()
plt.cla()
fontsize = 14
# fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (6,5),dpi = 300)
# sns.histplot(data = all_corr_frame,x = 'Normed PC1 Weight',y = 'Best Corr',ax = ax,bins = 30)
# joint = sns.jointplot(data=all_peak_info, x="Peak_Height", y="Peak_Width",marginal_kws=dict(bins=30),lw=0,s=3,alpha=0.8)
joint = sns.jointplot(data=all_peak_info, x="Peak_Height", y="Peak_Width",marginal_kws=dict(bins=25),legend=False,lw=0,s=3,alpha=0,ratio=3)
# marginal_kws=dict(bins=30)
# joint = sns.jointplot(data=all_corr_frame, x="Normed PC1 Weight", y="Best Corr", kind="hist")
sns.scatterplot(data=all_peak_info, x="Peak_Height", y="Peak_Width",hue='Loc',ax = joint.ax_joint,legend = False,lw=0,s=3,alpha=0.85)

joint.ax_marg_y.axhline(y=width_mid,linestyle='--',color = 'gray')
joint.ax_marg_x.axvline(x=height_mid,linestyle='--',color = 'gray')

joint.ax_joint.set_xlabel('')
joint.ax_joint.set_ylabel('')
joint.ax_joint.set_xlim(0,1)
joint.ax_joint.set_xticks([0,0.2,0.4,0.6,0.8,1])
joint.ax_joint.set_xticklabels([0,0.2,0.4,0.6,0.8,1],fontsize = fontsize)
joint.ax_joint.set_ylim(0,6)
joint.ax_joint.set_yticks(np.array([0,2,4,6])*1.301)
joint.ax_joint.set_yticklabels([0,2,4,6],fontsize = fontsize)
joint.figure.set_figwidth(6)
joint.figure.set_figheight(7)
joint.figure.set_dpi(300)


joint.savefig(ot.join(savepath,'FigS2CEF_Stats'),bbox_inches='tight')