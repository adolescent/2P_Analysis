'''
Answer the question of why we use low-pass filter, and what will happen if we don't filt.
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
# import umap
# import umap.plot
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
from Cell_Class.Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *



exp_path = r'D:\_All_Spon_Data_V1\L76_SM_Run03bug_210721'
ac = ot.Load_Variable_v2(exp_path,'Cell_Class.pkl')
sponrun =  ot.Load_Variable_v2(exp_path,'Spon_Before.pkl')
start = sponrun.index[0]
end = sponrun.index[-1]
all_cell_dic = ac.all_cell_dic
fps = ac.fps

savepath = r'D:\ZR\_Data_Temp\_Article_Data\_Revised_Data'

#%% #######################################################
# Generate new unfilted Z frames.
from Fix_Funcs import *
z_lp_on = Z_refilter(ac,start,end,'1-001',0.005,0.3)
z_lp_off = Z_refilter(ac,start,end,'1-001',0.005,False)
vmax = 4
vmin = -2

fig,ax = plt.subplots(ncols=1,nrows=2,figsize = (8,5),dpi=300,sharex=True)
sns.heatmap(z_lp_on[:,3000:3650],ax = ax[0],cbar=False,yticklabels=False,xticklabels=False,center = 0,vmax = vmax,vmin = vmin)
sns.heatmap(z_lp_off[:,3000:3650],ax = ax[1],cbar=False,yticklabels=False,xticklabels=False,center = 0,vmax = vmax,vmin = vmin)
ax[0].set_title('LP filter ON')
ax[1].set_title('LP filter OFF')

fig.tight_layout()


#%% #########################################################
# Freq analysis
freq_ticks,_,_,_,_ = FFT_Spectrum(z_lp_on[0,:],1.301)
freq_filt = np.zeros(shape = (len(ac),len(freq_ticks)),dtype='f8')
freq_unfilt = np.zeros(shape = (len(ac),len(freq_ticks)),dtype='f8')
for i in range(len(ac)):
    c_train = z_lp_on[i,:]
    _,c_power,_,_,_ = FFT_Spectrum(c_train,1.301,0.01,False)
    freq_filt[i,:] = c_power
    c_train = z_lp_off[i,:]
    _,c_power,_,_,_ = FFT_Spectrum(c_train,1.301,0.01,False)
    freq_unfilt[i,:] = c_power
    
freq_filt = pd.DataFrame(freq_filt[:,:-1],columns = freq_ticks[:-1])
freq_unfilt = pd.DataFrame(freq_unfilt[:,:-1],columns = freq_ticks[:-1])
#%% plot freq

vmax = 0.13
vmin = 0
fig,ax = plt.subplots(ncols=1,nrows=2,figsize = (3,6),dpi=300,sharex=True)
sns.heatmap(freq_filt,ax = ax[0],cbar=False,yticklabels=False,xticklabels=False,center = 0,vmax = vmax,vmin = vmin)
sns.heatmap(freq_unfilt,ax = ax[1],cbar=False,yticklabels=False,xticklabels=False,center = 0,vmax = vmax,vmin = vmin)
ax[0].set_title('LP filter ON')
ax[1].set_title('LP filter OFF')
ax[1].set_xticks([0,20,40,60])
ax[1].set_xticklabels([0,0.20,0.40,0.60])

fig.tight_layout()
#%% plot freq way 2
freq_unfilt_m = pd.melt(freq_unfilt,var_name='Freq',value_name='Power')
freq_filt_m = pd.melt(freq_filt,var_name='Freq',value_name='Power')


fig,ax = plt.subplots(ncols=1,nrows=2,figsize = (3,6),dpi=300,sharex=True,sharey=True)
sns.lineplot(data = freq_filt_m,x = 'Freq',y = 'Power',ax = ax[0])
sns.lineplot(data = freq_unfilt_m,x = 'Freq',y = 'Power',ax = ax[1])

ax[0].set_title('LP filter ON')
ax[1].set_title('LP filter OFF')
# ax[1].set_xticks([0,20,40,60])
ax[1].set_xticks([0,0.20,0.40,0.60])
ax[1].set_xticklabels([0,0.20,0.40,0.60])

#%% ###########################################
example_cell_on = z_lp_on[43,3000:3300]
example_cell_off = z_lp_off[43,3000:3300]
plt.plot(example_cell_on[150:250],alpha = 0.7)
plt.plot(example_cell_off[150:250],alpha = 0.7)

from scipy.signal import find_peaks,peak_widths

peaks_on, properties_on = find_peaks(example_cell_on, distance=3,height=-0.5) 
peaks_off, properties_off = find_peaks(example_cell_off, distance=3,height=-0.5) 
#%%
plt.plot(example_cell_off)
plt.plot(peaks_off,example_cell_off[peaks_off], "x")
plt.show()

plt.plot(example_cell_on)
plt.plot(peaks_on,example_cell_on[peaks_on], "x")
plt.show()