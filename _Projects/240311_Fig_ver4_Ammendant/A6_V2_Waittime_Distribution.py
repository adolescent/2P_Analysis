'''

This script will generate V2's wait time distribution, and we will compare it with V1.


Only orientation network's repeat is considered here.
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
from Cell_Class.UMAP_Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *


wp = r'D:\_Path_For_Figs\240312_Figs_v4_A1\A6_V2_Waittime_Disp'
# expt_loc = r'D:\_All_Spon_Data_V1\L76_18M_220902'

all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V2'))

#%% Functions will use
def Start_Time_Finder(series):
    # Find the indices where the series switches from 0 to 1
    switch_indices = np.where(np.diff(series) == 1)[0] + 1 
    return switch_indices

#%% ####################### STEP0 - SAVE ALL REPEAT CURVES ###########################
pcnum = 10
all_spon_repeats = {}
for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    c_ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = np.array(ot.Load_Variable(cloc,'Spon_Before.pkl'))
    # Use spontaneous embedded pca model.
    spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=c_spon,sample='Frame',pcnum=pcnum)
    # orientation trains
    analyzer_orien = UMAP_Analyzer(ac = c_ac,umap_model=spon_models,spon_frame=c_spon,od = 0, color = 0,orien = 1)
    analyzer_orien.Train_SVM_Classifier()
    spon_label_orien = analyzer_orien.spon_label
    # save all 3 trains into a pd frame.
    train_series = pd.DataFrame(spon_label_orien.T,columns = ['Orien']).T
    all_spon_repeats[cloc_name] = train_series

ot.Save_Variable(wp,'All_Repeat_Series_V2',all_spon_repeats)

#%%##### 1. Wait time of Each Network & elapse time of each networks.#####
# example_frame = (all_spon_repeats['L76_18M_220902']>0).astype('f8')
# od_start_time = Start_Time_Finder(np.array(example_frame.loc['OD',:]))
# orien_start_time = Start_Time_Finder(np.array(example_frame.loc['Orien',:]))
# color_start_time = Start_Time_Finder(np.array(example_frame.loc['Color',:]))

# get nearest next network waittime.
all_network_waittime = pd.DataFrame(columns = ['Loc','Net_Before','Net_After','Waittime','Start_Time'])
for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    c_repeat_frame = all_spon_repeats[cloc_name]
    binned_repeat_frame = (c_repeat_frame>0).astype('f8')
    c_orien = Start_Time_Finder(np.array(binned_repeat_frame.loc['Orien',:]))
    # then save each network's repeat by tuning.
    for j,c_start in enumerate(c_orien):
        if j < len(c_orien)-1:
            all_network_waittime.loc[len(all_network_waittime),:] = [cloc_name,'Orien','Orien',c_orien[j+1]-c_orien[j],c_start]

ot.Save_Variable(wp,'All_Network_Waittime_v2',all_network_waittime)

#%% ######################### PLOT 3 Waittime Disp Graphs ############################
def Weibul_Fit_Plotter(ax,disp,x_max):
    #fit
    params = stats.exponweib.fit(disp,floc = 0)
    # params = stats.weibull_min.fit(disp,floc = 0)
    x = np.linspace(0, x_max, 200)
    pdf_fitted = stats.exponweib.pdf(x, *params)
    # plot
    ax.hist(disp, bins=20, density=True, alpha=1,range=[0, x_max])
    ax.plot(x, pdf_fitted, 'r-', label='Fitted')
    ax.set_xlim(0,x_max)

    # calculate r2 at last,using QQ Plot method
    _,(slope, intercept, r) = stats.probplot(disp, dist=stats.exponweib,sparams = params,plot=None, rvalue=True)
    r2 = r**2
    return ax,params,r2

waittime_mean_frame = pd.DataFrame(columns = ['Start_Net','End_Net','Mean','std','R2'])

plt.clf()
plt.cla()
all_locs = list(set(all_network_waittime['Loc']))
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,6),dpi = 180, sharex='col',sharey='row')
vmax = [100,60,80]

for i,c_loc in enumerate(all_locs):
        c_waittime_disp = all_network_waittime.groupby('Loc').get_group(c_loc)['Waittime']
        # c_waittime_disp = all_network_waittime.loc[c_start&c_end,:]
        axes[i],_,c_r2 = Weibul_Fit_Plotter(axes[i],np.array(c_waittime_disp).astype('f8'),vmax[i])

        axes[i].text(vmax[i]*0.6,0.07,f'R2 = {c_r2:.3f}')
        axes[i].text(vmax[i]*0.6,0.06,f'n = {len(c_waittime_disp)}')
        axes[i].text(vmax[i]*0.6,0.05,f'Mean = {c_waittime_disp.mean():.4f}')
        axes[i].set_title(c_loc)
        axes[i].set_xlabel('Wait Time')


fig.suptitle('V2 Orientation Repeat Wait Time',size = 18,y = 0.97)
axes[0].set_ylabel('Prob. Density')
# axes[2].legend()
# axes[2,0].set_xlabel('To OD',size = 12)
# axes[2,1].set_xlabel('To Orientation',size = 12)
# axes[2,2].set_xlabel('To Color',size = 12)
# axes[0,0].set_ylabel('From OD',size = 12,rotation = 90)
# axes[1,0].set_ylabel('From Orientation',size = 12)
# axes[2,0].set_ylabel('From Color',size = 12)

fig.tight_layout()

#%% ############################### ANIMAL COMPARE ####################################
all_spon_repeats_v1 = ot.Load_Variable(r'D:\_Path_For_Figs\240312_Figs_v4_A1\A1_Waittime_Distribution\All_Repeat_Series.pkl')
all_spon_repeats_v2 = all_spon_repeats
all_v1_loc_names = list(all_spon_repeats_v1.keys())
all_v2_loc_names = list(all_spon_repeats_v2.keys())

L76_v1_loc = all_v1_loc_names[:4]
L85_v1_loc = all_v1_loc_names[4:6]
L76_v2_loc = all_v2_loc_names[:2]
L85_v2_loc = all_v2_loc_names[2]
all_orien_repeat_frame = pd.DataFrame(columns = ['Animal','Freq','Count','Brain_Area'])
# save V1
for i,c_v1_loc in enumerate(all_v1_loc_names[:6]):
     c_v1_orien_series = np.array(all_spon_repeats_v1[c_v1_loc].loc['Orien',:])>0
     c_v1_frames = c_v1_orien_series.sum()/len(c_v1_orien_series)
     c_v1_freq = Event_Counter(c_v1_orien_series)*1.301/len(c_v1_orien_series)
     if c_v1_loc in L76_v1_loc:
          all_orien_repeat_frame.loc[len(all_orien_repeat_frame),:] = ['L76',c_v1_freq,c_v1_frames,'V1']
     elif c_v1_loc in L85_v1_loc:
          all_orien_repeat_frame.loc[len(all_orien_repeat_frame),:] = ['L85',c_v1_freq,c_v1_frames,'V1']
# save V2
for i,c_v2_loc in enumerate(all_v2_loc_names):
     c_v2_orien_series = np.array(all_spon_repeats_v2[c_v2_loc].loc['Orien',:])>0
     c_v2_frames = c_v2_orien_series.sum()/len(c_v2_orien_series)
     c_v2_freq = Event_Counter(c_v2_orien_series)*1.301/len(c_v2_orien_series)
     if c_v2_loc in L76_v2_loc:
          all_orien_repeat_frame.loc[len(all_orien_repeat_frame),:] = ['L76',c_v2_freq,c_v2_frames,'V2']
     elif c_v2_loc in L85_v2_loc:
          all_orien_repeat_frame.loc[len(all_orien_repeat_frame),:] = ['L85',c_v2_freq,c_v2_frames,'V2']
#%% Plot Bar Plots
plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5),dpi = 180, sharex='col',sharey='row')
sns.barplot(data = all_orien_repeat_frame,x = 'Animal',y = 'Freq',hue = 'Brain_Area',ax = ax,capsize=.2,width=0.5)
# ax.legend(loc='upper right')
ax.legend(loc=(0.4,0.85))
ax.set_ylabel('Frequency (Hz)')
ax.set_title('Orientation Repeat Frequency',size = 16)