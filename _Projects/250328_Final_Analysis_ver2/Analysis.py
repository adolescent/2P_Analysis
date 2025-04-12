'''
This script will do simle examples for thesis graphs.

Just make it quick = =

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


warnings.filterwarnings("ignore")

expt_folder = r'D:\_DataTemp\_Fig_Datas\_All_Spon_Data_V1\L76_18M_220902'
# savepath = r'D:\_GoogleDrive_Files\#卒论\Figs'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
sponrun = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
start = sponrun.index[0]
end = sponrun.index[-1]
# ac.wp = savepath

#%%
'''
Fig 3-1, Getting color tuning map.
'''
# green_resp = ac.Color_t_graphs['Green-White'].loc['A_response']-ac.Color_t_graphs['Green-White'].loc['B_response']
# red_resp = ac.Color_t_graphs['Red-White'].loc['A_response']-ac.Color_t_graphs['Red-White'].loc['B_response']
# blue_resp = ac.Color_t_graphs['Blue-White'].loc['A_response']-ac.Color_t_graphs['Blue-White'].loc['B_response']

red_resp = ac.Color_t_graphs['Red-White'].loc['CohenD']
green_resp = ac.Color_t_graphs['Green-White'].loc['CohenD']
blue_resp = ac.Color_t_graphs['Blue-White'].loc['CohenD']

r = ac.Generate_Weighted_Cell(red_resp)
r = r.clip(-5,5)
r= (r*255/r.max()).astype('u1')
# r = (r.clip(-2.5,2.5)*255/2.5).astype('u1')
g = ac.Generate_Weighted_Cell(green_resp)
g = g.clip(-5,5)
g= (g*125/g.max()).astype('u1')
# g = (g.clip(-2.5,2.5)*255/2.5).astype('u1')
b = ac.Generate_Weighted_Cell(blue_resp)
b = b.clip(-5,5)
b= (b*255/b.max()).astype('u1')
# b = (b.clip(-2.5,2.5)*255/2.5).astype('u1')

graph = np.zeros(shape = (512,512,3),dtype='u1')
graph[:,:,0] = r
# graph[:,:,1] = g
graph[:,:,2] = b

plt.imshow(graph)
# sns.heatmap(graph,square=True,yticklabels=False,xticklabels=False,center = 0,cbar=False)


#%%
'''
Fig 3 Specturm of Stim-ON frame.
'''

all_dff = ot.Load_Variable_v2(r'G:\我的云端硬盘\#Figs\#250211_Revision1\Fig1\1e_All_Cell_dFF.pkl')

orien_series = Z_refilter(ac,ac.orienrun,0,99999).T
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
orien_freqs = Transfer_Into_Freq(orien_series)

plt.clf()
plt.cla()
vmax = 0.1
vmin = 0

fontsize = 14
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(4,5),dpi = 180,sharex= True)
# cbar_ax = fig.add_axes([0.97, .6, .02, .15])
sns.heatmap(orien_freqs[:60,:].T,center = 0,vmax=vmax,ax = ax[0],cbar=False,xticklabels=False,yticklabels=False,cmap = 'bwr')
# sns.heatmap(spon_freqs[:40,:].T,center = 0,vmax=0.15,ax = ax,cbar_ax= cbar_ax,xticklabels=False,yticklabels=False,cbar_kws={'label': 'Spectral Density'})
#plot global powers.
plotable_power = pd.DataFrame(orien_freqs[:60,:].T).melt(var_name='Freq',value_name='Prop.')
sns.lineplot(data = plotable_power,x='Freq',y='Prop.',ax = ax[1])
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_xticks([0,20,40,60])
ax[1].set_xticklabels([0,0.2,0.4,0.6],fontsize = fontsize)
ax[1].set_yticks([0.1,0.2])
ax[1].set_yticklabels([0.1,0.2],fontsize = fontsize)
ax[1].set_ylabel('')
ax[1].set_xlabel('')

fig.tight_layout()
#%%
'''

Fig 3 compare ISI and spon.

'''

ac_strength = ot.Load_Variable_v2(r'G:\我的云端硬盘\#Figs\#250211_Revision1\Fig1\1e_All_Cell_dFF_blank.pkl')

plotable_data = ac_strength
plt.clf()
plt.cla()
fontsize = 14

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4,3),dpi = 300,sharex= False)
fig.subplots_adjust(hspace=0.4)

pivoted_df = plotable_data.pivot(index=['Loc', 'Cell'], columns='In_Run', values=['dFF'])
pivoted_df = pivoted_df['dFF']
axes.plot([0,2],[0,2],color = 'gray', linestyle = '--')
scatter = sns.scatterplot(data=pivoted_df,x = 'Spontaneous',y = 'Stimulus_Blank',s = 3,ax = axes,linewidth = 0,alpha = 0.8,legend=False)
axes.set_xlim(0,2)
axes.set_ylim(0,2)

# axes[1].title.set_text('Cell dF/F Distribution')
# axes[0].set_xlabel('Spontaneous dF/F')
# axes[0].set_ylabel('Stimulus ON dF/F')
# axes[0].xaxis.tick_top()
# axes[0].xaxis.set_label_position('top') 

axes.set_yticks([0,0.5,1,1.5,2])
axes.set_yticklabels([0,0.5,1,1.5,2],fontsize = fontsize)
axes.set_xticks([0,0.5,1,1.5,2])
axes.set_xticklabels([0,0.5,1,1.5,2],fontsize = fontsize)


axes.set_xlabel('')
axes.set_ylabel('')

#%%
# Ratio

cell_num = len(ac_strength)//2
spon_stim_ratio = np.zeros(cell_num)
for i in range(cell_num):
    c_spon = ac_strength.iloc[2*i,-1]
    c_stim = ac_strength.iloc[2*i+1,-1]
    spon_stim_ratio[i] = c_spon/c_stim

# plot part
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,3),dpi = 300,sharex= False)

ax.axvline(x = spon_stim_ratio.mean(),linestyle='--',color = [0.7,0.7,0.7])
sns.histplot(spon_stim_ratio,bins = np.linspace(0,2.5,26),ax = ax)

ax.set_xlim(0,2.1)
ax.set_yticks([0,300,600,900])
ax.set_yticklabels([0,300,600,900],fontsize = fontsize)
ax.set_xticks([0,0.5,1,1.5,2])
ax.set_xticklabels([0,0.5,1,1.5,2],fontsize = fontsize)
ax.set_ylabel('')

#%%
'''
Fig 3-16 Pairwise corr analysis
'''

all_corr = ot.Load_Variable_v2(r'G:\我的云端硬盘\#Figs\#250211_Revision1\Fig4\All_Pair_Corrs.pkl')
all_loc = list(all_corr.keys())
for i,cloc in enumerate(all_loc):
    if i == 0:
        all_corr_concat = copy.deepcopy(all_corr[cloc])
    else:
        all_corr_concat = pd.concat((all_corr_concat,all_corr[cloc]))

#%% Plot dist correlations.
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.lines as mlines
def exponential_model(dist,A,  B, C):
    return A*np.exp(-B * dist) + C


# plt.style.use('dark_background')
plt.style.use('default')
fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (4,3),dpi=300)
rs = []
for i in range(8):
    example_corr = all_corr[all_loc[i]]
    params, covariance = curve_fit(
        exponential_model, 
        example_corr['Dist'], 
        example_corr['Corr'], 
        p0=[1,0.1,0.3]
    )
    A, B, C = params
    predicted_corr = exponential_model(example_corr['Dist'], A, B, C)
    r_squared = r2_score(example_corr['Corr'], predicted_corr)
    rs.append(r_squared)
    print(f'r2={r_squared}')
    # plot scatter and fitted graph
    # sns.scatterplot(data=example_corr,x='Dist',y='Corr',s=2,alpha=0.8,ax=ax,lw=0)
    x = np.arange(600)
    y = exponential_model(x,*params)
    ax.plot(x,y)
    pix_dist = 830/512
    decay_pix = np.log(2)/params[1]
    est_height = exponential_model(decay_pix,*params)
    # ax.axvline(x=decay_pix,ymin=0,ymax=est_height,linestyle='--',color='gray')
    # line = mlines.Line2D([decay_pix,decay_pix], [0,est_height], label='5%', color='r', linestyle='--', linewidth=1)
    # ax.add_line(line)
    # ax.set_ylabel('')
    # ax.set_xlabel('')

    ax.set_ylim(0.1,0.6)
    ax.set_xlim(0,600)
    # # # ax.set_ylim(0,1)
    ax.set_xticks([0/pix_dist,300/pix_dist,600/pix_dist,900/pix_dist])

    ax.set_xticklabels([0,300,600,900],fontsize = 12)
    ax.set_yticks([0.2,0.4,0.6])
    ax.set_yticklabels([0.2,0.4,0.6],fontsize = 12)
#%%
'''

Fig 3-17 V2 PCA basics.

'''
all_path_dic_v2 = list(ot.Get_Subfolders(r'D:\_DataTemp\_Fig_Datas\_All_Spon_Data_V2'))

all_pc_var = np.zeros(shape = (10,len(all_path_dic_v2)))

for i,cloc in enumerate(all_path_dic_v2):
# cloc = all_path_dic_v2[0]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    sponrun = ot.Load_Variable(cloc,'Spon_Before.pkl')
    start = sponrun.index[0]
    end = sponrun.index[-1]
    spon_series = Z_refilter(ac,'1-001',start,end).T
    spon_series = np.array(spon_series)
    pcnum = 10
    # real spon models
    spon_pcs,spon_coords,spon_model = Z_PCA(Z_frame=spon_series,sample='Frame',pcnum=pcnum)
    model_var_ratio = np.array(spon_model.explained_variance_ratio_)
    print(f'{pcnum} PCs explain Spontaneous VAR {model_var_ratio[:pcnum].sum()*100:.1f}%')
    all_pc_var[:,i] = spon_model.explained_variance_ratio_

#%% Plot explained var
plotable = pd.DataFrame(all_pc_var.T).melt(var_name='PC',value_name='Explained VAR Ratio')
plotable['Explained VAR Ratio'] = plotable['Explained VAR Ratio']*100
plotable['PC'] = plotable['PC']+1

plt.clf()
plt.cla()
fontsize = 14

fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (5,4),dpi = 300)
sns.barplot(data = plotable,y = 'Explained VAR Ratio',x = 'PC',ax = ax,capsize=0.2)
# ax.set_xlabel('PC',size = 12)
# ax.set_ylabel('Explained Ratio(%)',size = 12)
# ax.set_title('Each PC explained Variance',size = 14)
ax.set_ylim(0,32)
top10_sum = all_pc_var.sum(0)
ax.set_yticks([0,10,20,30])
ax.set_yticklabels([0,10,20,30],fontsize = fontsize)
ax.set_xticks(np.arange(0,10))
ax.set_xticklabels(np.arange(1,11),fontsize = fontsize)

ax.set_ylabel('')
ax.set_xlabel('')
print(f'Top 10 PC explain VAR={top10_sum.mean():.4f}, std={top10_sum.std():.4f}')


#%% Top PC10
plt.clf()
plt.cla()
vmax = 0.1
vmin = -0.1
font_size = 13
fig,axes = plt.subplots(nrows=2, ncols=5,figsize = (12,6),dpi = 300)
# cbar_ax = fig.add_axes([1, .45, .01, .2])
for i in tqdm(range(10)):
    c_pc = spon_pcs[i,:]
    c_pc_graph = ac.Generate_Weighted_Cell(c_pc)
    # sns.heatmap(c_pc_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[i//5,i%5],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True,cmap = cmaps.pinkgreen_light)
    sns.heatmap(c_pc_graph,center = 0,xticklabels=False,yticklabels=False,ax = axes[i//5,i%5],vmax = vmax,vmin = vmin,cbar= False,square=True,cmap = 'gist_gray')
    # axes[i//5,i%5].set_title(f'PC {i+1}',size = font_size)
fig.tight_layout()
# fig.savefig(ot.join(savepath,'Fig2D_PC_Comps.png'),bbox_inches='tight')
#%% similar to func map.
# get all response
# od_resp = ac.OD_t_graphs['OD'].loc['CohenD',:]
hv_resp = ac.Orien_t_graphs['H-V'].loc['t_value',:]
ao_resp = ac.Orien_t_graphs['A-O'].loc['t_value',:]
red_resp = ac.Color_t_graphs['Red-White'].loc['t_value',:]
blue_resp = ac.Color_t_graphs['Blue-White'].loc['t_value',:]
all_response = [hv_resp,ao_resp,red_resp,blue_resp]

#and generate data frame
pc_list = ['PC{}'.format(i) for i in range(1, 11)]
networks = ['HV','AO','Red','Blue']
all_corrs = pd.DataFrame(0.0,columns = pc_list,index = networks)

# fill it with pearsonr
for i,c_pc in enumerate(pc_list):
    c_pc_response = spon_pcs[i,:]
    for j,c_net in enumerate(networks):
        c_stim_response = all_response[j]
        c_r,_ = stats.pearsonr(c_pc_response,c_stim_response)
        all_corrs.loc[c_net,c_pc] = abs(c_r)
#%% plot it
value_max = 0.8
value_min = 0
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (8,4),dpi = 300)
sns.heatmap(all_corrs,center = 0,annot=True,cmap = 'bwr', fmt=".2f",ax = ax,vmax = value_max,vmin = value_min,cbar=False,annot_kws={"size": 14})

# ax.set_xticks([])
# ax.set_yticks([])
ax.set_xticks(np.arange(0,10)+0.5)
ax.set_xticklabels(np.arange(1,11),fontsize = 14)
ax.set_yticklabels(['0°-90°','45°-135°','Red','Blue'],fontsize = 14)
ax.xaxis.tick_top()

plt.show()