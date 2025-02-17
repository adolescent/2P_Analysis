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

warnings.filterwarnings("ignore")

expt_folder = r'D:\_DataTemp\_Fig_Datas\_All_Spon_Data_V1\L76_18M_220902'
savepath = r'G:\我的云端硬盘\#Figs\#250211_Revision1\Fig1'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
sponrun = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
start = sponrun.index[0]
end = sponrun.index[-1]

# generate new spon series, remove LP filter.
# NOTE different shape!
spon_series = Z_refilter(ac,'1-001',start,end).T

#%%
'''
Fig 1D/E/F, we generate original Heatmaps, without annotating.
'''

# get spon,stim,shuffle frames.
orien_series = Z_refilter(ac,ac.orienrun,0,99999).T
spon_shuffle = Spon_Shuffler(spon_series,method='phase',filter_para=(0.005,0.65))
# transfer them into pd frame, for further process.
spon_series = pd.DataFrame(spon_series,columns = ac.acn,index = range(len(spon_series)))
orien_series = pd.DataFrame(orien_series,columns = ac.acn,index = range(len(orien_series)))
spon_shuffle_frame = pd.DataFrame(spon_shuffle,columns = ac.acn,index = range(len(spon_shuffle)))

# Sort Orien By Cells actually we sort only by raw data.
rank_index = pd.DataFrame(index = ac.acn,columns=['Best_Orien','Sort_Index','Sort_Index2'])
for i,cc in enumerate(ac.acn):
    rank_index.loc[cc]['Best_Orien'] = ac.all_cell_tunings[cc]['Best_Orien']
    if ac.all_cell_tunings[cc]['Best_Orien'] == 'False':
        rank_index.loc[cc]['Sort_Index']=-1
        rank_index.loc[cc]['Sort_Index2']=0
    else:
        orien_tunings = float(ac.all_cell_tunings[cc]['Best_Orien'][5:])
        # rank_index.loc[cc]['Sort_Index'] = np.sin(np.deg2rad(orien_tunings))
        rank_index.loc[cc]['Sort_Index'] = orien_tunings
        rank_index.loc[cc]['Sort_Index2'] = np.cos(np.deg2rad(orien_tunings))
# actually we sort only by raw data.
sorted_cell_sequence = rank_index.sort_values(by=['Sort_Index'],ascending=False)
# and we try to reindex data.
sorted_stim_response = orien_series.T.reindex(sorted_cell_sequence.index).T
sorted_spon_response = spon_series.T.reindex(sorted_cell_sequence.index).T
sorted_shuffle_response = spon_shuffle_frame.T.reindex(sorted_cell_sequence.index).T
#%% Plot Cell Stim Maps
plt.clf()
plt.cla()

# plot bar seperately
vmax = 4
vmin = -2
fig_bar,ax_bar = Cbar_Generate(vmin,vmax)
# fig_bar.savefig(ot.join(savepath,'Fig1FGH_Bars.png') ,bbox_inches='tight')
#%% Plot no bar graphs
plt.clf()
plt.cla()
label_size = 10
## raw graph
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10,8),dpi = 240)
sns.heatmap((sorted_stim_response .iloc[1000:1650,:].T),center = 0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = vmax,vmin = vmin,cbar= False,cmap = 'bwr')
sns.heatmap(sorted_spon_response.iloc[4700:5350,:].T,center = 0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = vmax,vmin = vmin,cbar= False,cmap = 'bwr')
sns.heatmap(sorted_shuffle_response.iloc[4700:5350,:].T,center = 0,xticklabels=False,yticklabels=False,ax = axes[2],vmax = vmax,vmin = vmin,cbar= False,cmap = 'bwr')
## annotate rectangle
from matplotlib.patches import Rectangle
axes[1].add_patch(Rectangle((175,0), 6, 520, fill=False, edgecolor='blue', lw=1,alpha = 0.8))
axes[0].add_patch(Rectangle((461,0), 6, 520, fill=False, edgecolor='blue', lw=1,alpha = 0.8))
axes[2].add_patch(Rectangle((461,0), 6, 520, fill=False, edgecolor='blue', lw=1,alpha = 0.8))
## label
fps = 1.301
axes[2].set_xticks([0*fps,100*fps,200*fps,300*fps,400*fps,500*fps])
axes[2].set_xticklabels([0,100,200,300,400,500],fontsize = label_size)

# fig.tight_layout()
# plt.show()
# fig.savefig(ot.join(savepath,'Fig1FGH_Heatmap.png'), bbox_inches='tight')

#%%
'''
Fig I/J/K, recover example response in box.
'''
stim_start_point = 175
spon_start_point = 461

stim_recover = orien_series.loc[1000+stim_start_point:1000+stim_start_point+6].mean(0)
spon_recover = spon_series.loc[4700+spon_start_point:4700+spon_start_point+6].mean(0)
shuffle_recover = spon_shuffle_frame.loc[0+spon_start_point:0+spon_start_point+6].mean(0)
stim_recover_map = ac.Generate_Weighted_Cell(stim_recover)
spon_recover_map = ac.Generate_Weighted_Cell(spon_recover)
shuffle_recover_map = ac.Generate_Weighted_Cell(shuffle_recover)
#%% Plot colorbar seperately
plt.clf()
plt.cla()
vmax = 3
vmin = -2
fig_bar,ax_bar = Cbar_Generate(vmin,vmax,cmap=None)
fig_bar.savefig(ot.join(savepath,'Fig1IJK_Bars.png'),bbox_inches='tight')
#%% main part
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5,8),dpi = 180)
fig.tight_layout()
sns.heatmap(spon_recover_map,center=0,xticklabels=False,yticklabels=False,ax = axes[0],vmax = vmax,vmin = vmin,cbar= False,square=True)
sns.heatmap(stim_recover_map,center = 0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = vmax,vmin = vmin,cbar= False,square=True)
sns.heatmap(shuffle_recover_map,center=0,xticklabels=False,yticklabels=False,ax = axes[2],vmax = vmax,vmin = vmin,cbar= False,square=True)


# axes[0].set_title('Spontaneous Response',size = 11)
# axes[1].set_title('Stim-induced Response',size = 11)
# axes[2].set_title('Shuffled Response',size = 11)
fig.tight_layout()
# fig.savefig(ot.join(savepath,'Fig1IJK_Recovered.png'), bbox_inches='tight')

plt.show()

#%%
'''
Fig L/M, This part is a little different, as we plot all power, not only 0.3Hz.
'''
def Transfer_Into_Freq(input_matrix,freq_bin = 0.01,fps = 1.301):
    input_matrix = np.array(input_matrix)
    # get raw frame spectrums.
    all_specs = np.zeros(shape = ((input_matrix.shape[0]// 2)-1,input_matrix.shape[1]),dtype = 'f8')
    for i in range(input_matrix.shape[1]):
        c_series = input_matrix[:,i]
        c_fft = np.fft.fft(c_series)
        power_spectrum = np.abs(c_fft)[1:input_matrix.shape[0]// 2] ** 2
        power_spectrum = power_spectrum/power_spectrum.sum()
        all_specs[:,i] = power_spectrum
    
    binnum = int(fps/(2*freq_bin))
    binsize = round(len(all_specs)/binnum)
    binned_freq = np.zeros(shape = (binnum,input_matrix.shape[1]),dtype='f8')
    for i in range(binnum):
        c_bin_freqs = all_specs[i*binsize:(i+1)*binsize,:].sum(0)
        binned_freq[i,:] = c_bin_freqs
    return binned_freq

spon_freqs = Transfer_Into_Freq(spon_series)
orien_freqs = Transfer_Into_Freq(orien_series)

#%% Plot spectrum
# still bar first
plt.clf()
plt.cla()
vmax = 0.1
vmin = 0
fig_bar,ax_bar = Cbar_Generate(vmin,vmax,cmap='bwr',aspect=5)
# fig_bar.savefig(ot.join(savepath,'Fig1L_Bars.png'),bbox_inches='tight')
#%% main plot here

plt.cla()
plt.clf()
fontsize = 14
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(4,5),dpi = 180,sharex= True)
# cbar_ax = fig.add_axes([0.97, .6, .02, .15])
sns.heatmap(spon_freqs[:60,:].T,center = 0,vmax=vmax,ax = ax[0],cbar=False,xticklabels=False,yticklabels=False,cmap = 'bwr')
# sns.heatmap(spon_freqs[:40,:].T,center = 0,vmax=0.15,ax = ax,cbar_ax= cbar_ax,xticklabels=False,yticklabels=False,cbar_kws={'label': 'Spectral Density'})
#plot global powers.
plotable_power = pd.DataFrame(spon_freqs[:60,:].T).melt(var_name='Freq',value_name='Prop.')
sns.lineplot(data = plotable_power,x='Freq',y='Prop.',ax = ax[1])
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_xticks([0,20,40,60])
ax[1].set_xticklabels([0,0.2,0.4,0.6],fontsize = fontsize)
ax[1].set_yticks([0.03,0.06])
ax[1].set_yticklabels([0.03,0.06],fontsize = fontsize)
ax[1].set_ylabel('')
ax[1].set_xlabel('')

fig.tight_layout()
fig.savefig(ot.join(savepath,'Fig1L_FFT_Power_rescale.png'),bbox_inches='tight')