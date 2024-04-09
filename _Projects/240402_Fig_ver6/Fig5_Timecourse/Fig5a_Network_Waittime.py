'''
Calculate each network's repeat waittime.
It's a perfect fit of weibull distribution

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


wp = r'D:\_Path_For_Figs\240312_Figs_v4_A1\A1_Waittime_Distribution'
expt_loc = r'D:\_All_Spon_Data_V1\L76_18M_220902'

all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)

all_spon_repeats = ot.Load_Variable(wp,'All_Repeat_Series.pkl')
#%% Functions will use
def Start_Time_Finder(series):
        
    # Find the indices where the series switches from 0 to 1
    switch_indices = np.where(np.diff(series) == 1)[0] + 1 


    return switch_indices

#%% ######################## Fig5a Show of How to get waittime ##############
example_repeat = all_spon_repeats['L76_18M_220902']
example_repeat_runs = example_repeat.loc['Orien',:]>0
plot_range = (2300,2330)

from matplotlib.patches import Rectangle
fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(7,3),dpi = 180)

for i, value in enumerate(example_repeat_runs[plot_range[0]:plot_range[1]]):
    # If the value is True, plot a filled rectangle
    if value:
        print(i)
        rect = Rectangle((i+plot_range[0], 0), 1, 1, facecolor='blue', edgecolor='blue')
        ax.add_patch(rect)
ax.set_xlim(plot_range[0]-3,plot_range[1]+3)
ax.set_ylim(0,1.8)
ax.set_title
ax.axes.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_title('Orientation Repeats in Spontaneous Activity')
ax.set_xticks((np.arange(1770,1800,10)*1.301).astype('i4'))
ax.set_xticklabels(np.arange(1770,1800,10))
ax.set_xlabel('Time (s)')

plt.show()

#%% ######################## Fig5b Waittime Plot ##############
from sklearn.metrics import r2_score

# all_network_waittime = ot.Load_Variable(wp,'All_Network_Waittime.pkl')
all_network_waittime_all_animal = ot.Load_Variable(wp,'All_Network_Waittime.pkl')
all_network_waittime_all_animal['Animal'] = 'Undefined'
all_locname = list(set(all_network_waittime_all_animal['Loc']))
all_locname.sort()
L76_locs = all_locname[:4]
L85_locs = all_locname[4:6]
L91_locs = all_locname[6:]
for i in tqdm(range(len(all_network_waittime_all_animal))):
    c_loc = all_network_waittime_all_animal.loc[i,'Loc']
    if c_loc in L76_locs:
        all_network_waittime_all_animal.loc[i,'Animal'] = 'L76'
    elif c_loc in L85_locs:
        all_network_waittime_all_animal.loc[i,'Animal'] = 'L85'
    elif c_loc in L91_locs:
        all_network_waittime_all_animal.loc[i,'Animal'] = 'L91'
#%% all start times
all_network_waittime = all_network_waittime_all_animal
all_network_waittime['Waittime'] = all_network_waittime['Waittime']/1.301

OD_starts = all_network_waittime['Net_Before'] == 'OD'
Orien_starts = all_network_waittime['Net_Before'] == 'Orien'
Color_starts = all_network_waittime['Net_Before'] == 'Color'
# all end times 
OD_ends = all_network_waittime['Net_After'] == 'OD'
Orien_ends = all_network_waittime['Net_After'] == 'Orien'
Color_ends = all_network_waittime['Net_After'] == 'Color'


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
#%%
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,5),dpi = 180, sharex='col',sharey='row')
network_seq = ['OD','Orien','Color']
vmax = [80,45,70]

od_waittime_disp = all_network_waittime.loc[OD_starts&OD_ends,'Waittime']
orien_waittime_disp = all_network_waittime.loc[Orien_starts&Orien_ends,'Waittime']
color_waittime_disp = all_network_waittime.loc[Color_starts&Color_ends,'Waittime']
for i,c_data in enumerate([od_waittime_disp,orien_waittime_disp,color_waittime_disp]):
    c_median = c_data.median()
    axes[i].axvline(x = c_median,color = 'gray',linestyle = '--')
    axes[i],_,c_r2 = Weibul_Fit_Plotter(axes[i],np.array(c_data).astype('f8'),vmax[i])
    axes[i].text(vmax[i]*0.6,0.07,f'R2 = {c_r2:.3f}')
    axes[i].text(vmax[i]*0.6,0.06,f'N repeat = {len(c_data)}')
    axes[i].text(vmax[i]*0.6,0.05,f'Median = {c_median:.2f}')
    axes[i].set_title(network_seq[i],size = 14)

fig.suptitle('Network Waittime',size = 20,y = 0.97)
# axes[0].legend()
axes[0].set_ylabel('Prob. Density',size = 14)
axes[1].set_xlabel('Waittime (s)',size = 14)

fig.tight_layout()