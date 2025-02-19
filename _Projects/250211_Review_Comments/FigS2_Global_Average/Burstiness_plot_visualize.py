'''
After burstiness generation, this script will plot and combine it.

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
from sklearn.model_selection import cross_val_score
from sklearn import svm
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
from Cell_Class.Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *
from Review_Fix_Funcs import *
from Filters import Signal_Filter_v2
import warnings

save_path = r'D:\_GoogleDrive_Files\#Figs\#250211_Revision1\FigS2'

burstiness = ot.Load_Variable_v2(save_path,'Burstiness_V1.pkl')
burstiness_s = ot.Load_Variable_v2(save_path,'Burstiness_V1_shuffle10.pkl')
burstiness_v2 = ot.Load_Variable_v2(save_path,'Burstiness_V2.pkl')
burstiness_s_v2 = ot.Load_Variable_v2(save_path,'Burstiness_V2_shuffle10.pkl')

#%% combine V1 and V2
burstiness['Area']='V1'
burstiness_v2['Area']='V2'
burstiness_s['Area']='V1'
burstiness_s_v2['Area']='V2'

burstiness_comb = pd.concat([burstiness,burstiness_v2])
burstiness_comb_s = pd.concat([burstiness_s,burstiness_s_v2])
burstiness_comb['Data_Type'] = 'Real'
burstiness_comb_s['Data_Type'] = 'Shuffle'

#%% plot part
plotable = pd.concat([burstiness_comb,burstiness_comb_s])

fontsize = 14
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(5,5),dpi=300)

sns.barplot(data = plotable,x = 'Area',y='Burstiness',hue='Data_Type',width=0.5,capsize=0.15,ax = ax,legend=False,lw=1)
ax.set_yticks([0,0.05,0.1])
ax.set_yticklabels([0,0.05,0.1],fontsize = fontsize)
ax.set_xticklabels(['V1','V2'],fontsize = fontsize)

ax.set_ylabel('')
ax.set_xlabel('')
fig.savefig(ot.join(save_path,'Burstiness_Compare.png'),bbox_inches = 'tight')

#%%
'''
Another plot, show waittime distribution of example cell location.
'''

waittime_v1 = ot.Load_Variable(save_path,'Waittime_V1.pkl')
waittime_v2 = ot.Load_Variable(save_path,'Waittime_V2.pkl')
#%% Plot and fit weibul
plotable = pd.concat([waittime_v1,waittime_v2])
plotable = plotable.groupby('Loc').get_group('L76_18M_220902')

def Weibul_Fit_Plotter(ax,disp,x_max):
    #fit
    params = stats.exponweib.fit(disp,floc = 0,method='mle')
    # params = stats.exponweib.fit(disp,loc=0,method='mle')
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
all_waittime = np.array(plotable['Waittime']).astype('f8')
c_median = np.median(all_waittime)
ax.axvline(x = c_median,color = 'gray',linestyle = '--')

ax,_,c_r2 = Weibul_Fit_Plotter(ax,all_waittime,vmax)
ax.text(vmax*0.6,0.07,f'R2 = {c_r2:.3f}')
ax.text(vmax*0.6,0.06,f'N repeat = {len(all_waittime)}')
ax.text(vmax*0.6,0.05,f'Median = {c_median/1.301:.3f} s')
ax.set_xticks(np.arange(0,50,10)*1.301)
ax.set_xticklabels(np.arange(0,50,10))

ax.set_title('Global Ensemble Waittime',size = 14)
# fig.savefig(ot.join(save_path,'All_Loc_Waittime.png'),bbox_inches='tight')
fig.savefig(ot.join(save_path,'All_Loc_Waittime.png'),bbox_inches='tight')

