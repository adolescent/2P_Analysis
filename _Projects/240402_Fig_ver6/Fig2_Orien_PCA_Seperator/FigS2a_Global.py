'''
Try to get global ensemble by threshold method.
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

import warnings
warnings.filterwarnings("ignore")

wp = r'D:\_All_Spon_Data_V1\L76_18M_220902'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
spon_series = np.array(ot.Load_Variable(wp,'Spon_Before.pkl'))
# if we need raw frame dF values
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
#%% Get averate responses.
thres = 2
avr_response = spon_series.mean(1)

#%% P1 verify viability of global ensemble with averaged response.
used_part = [4700,5350]
vmax = 5
vmin = -2
plt.clf()
plt.cla()

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,5),dpi = 180,sharex= True)
cbar_ax = fig.add_axes([1, 0.62, .01, .2])
sns.heatmap((spon_series[used_part[0]:used_part[1],:].T),center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = vmax,vmin = vmin,cbar_ax= cbar_ax)
plotable_avr = avr_response[used_part[0]:used_part[1]]
plotable_avr[plotable_avr<0] = 0
axes[1].plot(plotable_avr)

from scipy.signal import find_peaks
peaks, _ = find_peaks(plotable_avr, height=0,distance=3)
axes[1].plot(peaks, plotable_avr[peaks], "x")

axes[0].set_title('All Cell Response')
axes[1].set_title('Averaged Positive Response')
fig.tight_layout()

#%% P2 different thresed peak num and peat heights.

thres = [0,0.5,1,2]
all_peak_info = pd.DataFrame(columns = ['Thres','Loc','Peak_Freq'])
all_peak_height = pd.DataFrame(columns = ['Thres','Loc','Peak_Height'])


for j,c_thres in enumerate(thres):

    for i,cloc in tqdm(enumerate(all_path_dic)):
        cloc_name = cloc.split('\\')[-1]
        c_spon = np.array(ot.Load_Variable(cloc,'Spon_Before.pkl'))
        thres_avr = c_spon.mean(1)
        thres_avr[thres_avr<0] = 0 
        peaks, _ = find_peaks(thres_avr, height=c_thres,distance=3)
        peak_heights = thres_avr[peaks]
        c_freq = len(peaks)*1.301/len(thres_avr)
        # save into matrix.
        all_peak_info.loc[len(all_peak_info),:] = [c_thres,cloc_name,c_freq]
        for k,c_height in enumerate(peak_heights):
            all_peak_height.loc[len(all_peak_height ),:] = [c_thres,cloc_name,c_height]
#%% Plot Freq with thres.
plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (3,5),dpi = 180)
all_peak_info['Thres'] = all_peak_info['Thres'].astype('f8')
sns.boxplot(data = all_peak_info,hue = 'Thres',x = 'Thres',y = 'Peak_Freq',ax = ax,palette='tab10',showfliers = False)
ax.set_title('Global Ensemble Frequency')
ax.set_ylabel('Ensemble Frequency(Hz)')
ax.set_ylim(0,0.2)
#%% Plot Height with thres.
plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (3,5),dpi = 180)
ax.axhline(y = 0,color = 'gray',linestyle = '--')
ax.set_ylim(-0.5,4)
all_peak_height['Thres'] = all_peak_height['Thres'].astype('f8')
all_peak_height['Peak_Height'] = all_peak_height['Peak_Height'].astype('f8')
sns.boxenplot(data = all_peak_height,hue = 'Thres',x = 'Thres',y = 'Peak_Height',ax = ax,palette='tab10',width = 0.5,legend=False)
ax.set_title('Global Ensemble Strength')
ax.set_ylabel('Peak Heights')
# ax.legend(loc = 'lower right')
# ax.legend('')

#%% Compare global ensemble with weibull distribution.
waittime_thres = 0
waittime_frame = pd.DataFrame(columns = ['Loc','Waittime'])
for i,cloc in tqdm(enumerate(all_path_dic)):
    c_thres = waittime_thres
    cloc_name = cloc.split('\\')[-1]
    c_spon = np.array(ot.Load_Variable(cloc,'Spon_Before.pkl'))
    thres_avr = c_spon.mean(1)
    thres_avr[thres_avr<0] = 0 
    peaks, _ = find_peaks(thres_avr, height=c_thres,distance=3)
    peak_heights = thres_avr[peaks]
    for j in range(len(peaks)-1):
        c_waittime = peaks[j+1]-peaks[j]
        waittime_frame.loc[len(waittime_frame),:] = [cloc_name,c_waittime]
#%% Plot weibul fits.
def Weibul_Fit_Plotter(ax,disp,x_max):
    #fit
    params = stats.exponweib.fit(disp,floc = 0)
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
c_median = waittime_frame['Waittime'].median()
ax.axvline(x = c_median,color = 'gray',linestyle = '--')

ax,_,c_r2 = Weibul_Fit_Plotter(ax,waittime_frame['Waittime'].astype('f8'),vmax)
ax.text(vmax*0.6,0.07,f'R2 = {c_r2:.3f}')
ax.text(vmax*0.6,0.06,f'N repeat = {len(waittime_frame)}')
ax.text(vmax*0.6,0.05,f'Median = {c_median}')

ax.set_title('Global Ensemble Waittime',size = 14)

#%% Height compare with orien repeats.
Thres0_heights = all_peak_height[all_peak_height['Thres'] == 0.0]
global_spon_heights = pd.DataFrame(index = range(len(Thres0_heights)),columns=['Loc','Strength','Ensemble Type'])
global_spon_heights['Loc'] = Thres0_heights['Loc']
global_spon_heights['Strength'] = Thres0_heights['Peak_Height']
global_spon_heights['Ensemble Type'] = 'All Ensemble'


all_spon_heights = pd.DataFrame(columns = ['Loc','Strength','Ensemble Type'])
for i,cloc in enumerate(all_path_dic):
    cloc_name = cloc.split('\\')[-1]
    c_ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = np.array(ot.Load_Variable(cloc,'Spon_Before.pkl'))
    spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=c_spon,sample='Frame',pcnum=10)
    c_analyzer = UMAP_Analyzer(ac = c_ac,umap_model=spon_models,spon_frame=c_spon,od = 0,orien = 1,color = 0,isi = True)
    c_analyzer.Train_SVM_Classifier(C=1)
    spon_label = c_analyzer.spon_label
    all_events,_ = Label_Event_Cutter(spon_label>0)
    for j,c_event in tqdm(enumerate(all_events)):
        c_event_avrs = c_spon[c_event,:].mean(1)
        all_spon_heights.loc[len(all_spon_heights)] = [cloc_name,c_event_avrs.max(),'PCA Finded']
    # below is all repeat avr.
    # orien_repeats = c_spon[spon_label>0,:].mean(1)
    # for j,c_height in tqdm(enumerate(orien_repeats)):
    #     all_spon_heights.loc[len(all_spon_heights)] = [cloc_name,c_height,'PCA Finded']

comparable_frames = pd.concat([global_spon_heights,all_spon_heights])
#%% compare 2 graphs.

plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (3,5),dpi = 144)
sns.barplot(data = comparable_frames,y = 'Strength',x = 'Ensemble Type',hue = 'Ensemble Type',ax = ax,width = 0.5)
ax.set_xlabel('Ensemble Type',size = 10)
ax.set_ylabel('Mean Z',size = 10)
ax.set_title('Strength Compare of Ensembles',size = 12)
fig.tight_layout()