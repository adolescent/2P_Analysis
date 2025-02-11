'''
This 2 graph will compare shuffle, so we need stats here.

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
from Classifier_Analyzer import *

import warnings
warnings.filterwarnings("ignore")

# wp = r'D:\_All_Spon_Data_V1\L76_18M_220902'
# ac = ot.Load_Variable(wp,'Cell_Class.pkl')
# spon_series = ot.Load_Variable(wp,'Spon_Before.pkl')
savepath = r'D:\_Path_For_Figs\240614_Figs_ver_F2\Fig2'

all_path_dic = list(ot.Get_Subfolders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)


all_pc_var,all_pc_best_corr,pc1_corrs = ot.Load_Variable(savepath,'All_PC_Corr_Infos.pkl')
all_pc_var_s,all_pc_best_corr_s,pc1_corrs_s = ot.Load_Variable(savepath,'All_PC_Corr_Infos_Shuffle500.pkl')

#%%
'''
PCA Step1, get explained VAR of all locs all 10 pcs.
And get the explained VAR of all locs.
Compare figs: OD,HV,AO,red,green,blue.
'''
pcnum = 10
N_shuffle = 500
all_pc_var = np.zeros(shape = (pcnum,len(all_path_dic)))
all_pc_best_corr = np.zeros(shape = (pcnum,len(all_path_dic)))
pc1_corrs = np.zeros(len(all_path_dic))

all_pc_var_s = np.zeros(shape = (pcnum,len(all_path_dic)*N_shuffle))
all_pc_best_corr_s = np.zeros(shape = (pcnum,len(all_path_dic)*N_shuffle))
pc1_corrs_s = np.zeros(len(all_path_dic)*N_shuffle)

counter = 0
for i,cloc in enumerate(all_path_dic):
    ac = ot.Load_Variable(cloc,'Cell_Class.pkl')
    spon_series = np.array(ot.Load_Variable(cloc,'Spon_Before.pkl'))
    global_avr = spon_series.mean(1)
    spon_pcs,spon_coords,spon_model = Z_PCA(Z_frame=spon_series,sample='Frame',pcnum=pcnum)
    # record all PC's explained var.
    all_pc_var[:,i] = spon_model.explained_variance_ratio_
    pc1_corrs[i],_ = stats.pearsonr(global_avr,spon_coords[:,0])

    ##  and we compare it's correlation with stim maps
    # get all response
    od_resp = ac.OD_t_graphs['OD'].loc['CohenD',:]
    hv_resp = ac.Orien_t_graphs['H-V'].loc['CohenD',:]
    ao_resp = ac.Orien_t_graphs['A-O'].loc['CohenD',:]
    red_resp = ac.Color_t_graphs['Red-White'].loc['CohenD',:]
    blue_resp = ac.Color_t_graphs['Blue-White'].loc['CohenD',:]
    all_response = [od_resp,hv_resp,ao_resp,red_resp,blue_resp]
    # then calculate each pc's best corr.
    for j in range(pcnum):
        cpc = spon_pcs[j,:]
        networks = ['OD','HV','AO','Red','Blue']
        all_r = np.zeros(len(networks))
        for k,c_net in enumerate(networks):
            c_stim_response = all_response[k]
            c_r,_ = stats.pearsonr(cpc,c_stim_response)
            all_r[k] = abs(c_r)
        best_r = all_r.max()
        all_pc_best_corr[j,i] = best_r

    # shuffle here.
    for l in tqdm(range(N_shuffle)):
        spon_s = Spon_Shuffler(spon_series)
        spon_pcs_s,spon_coords_s,spon_model_s = Z_PCA(Z_frame=spon_s,sample='Frame',pcnum=pcnum)
        global_avr_s = spon_s.mean(1)
        pc1_corrs_s[i*N_shuffle+l],_ = stats.pearsonr(global_avr_s,spon_coords_s[:,0])
        all_pc_var_s[:,counter] = spon_model_s.explained_variance_ratio_

        for j in range(pcnum):
            cpc = spon_pcs_s[j,:]
            networks = ['OD','HV','AO','Red','Blue']
            all_r = np.zeros(len(networks))
            for k,c_net in enumerate(networks):
                c_stim_response = all_response[k]
                c_r,_ = stats.pearsonr(cpc,c_stim_response)
                all_r[k] = abs(c_r)
            best_r = all_r.max()
            all_pc_best_corr_s[j,counter] = best_r
        counter += 1

ot.Save_Variable(savepath,'All_PC_Corr_Infos',[all_pc_var,all_pc_best_corr,pc1_corrs])
ot.Save_Variable(savepath,'All_PC_Corr_Infos_Shuffle500',[all_pc_var_s,all_pc_best_corr_s,pc1_corrs_s])

#%% Generation done, plot part here.
'''
Fig 2C & S2C, Regenerate explained VAR graph, include error bars on it.
'''
plotable = pd.DataFrame(all_pc_var_s.T).melt(var_name='PC',value_name='Explained VAR Ratio')
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
ax.set_ylim(0,40)
top10_sum = all_pc_var.sum(0)
ax.set_yticks([0,10,20,30,40])
ax.set_yticklabels([0,10,20,30,40],fontsize = fontsize)
ax.set_xticks(np.arange(0,10))
ax.set_xticklabels(np.arange(1,11),fontsize = fontsize)

ax.set_ylabel('')
ax.set_xlabel('')

print(f'Top 10 PC explain VAR={top10_sum.mean():.4f}, std={top10_sum.std():.4f}')


#%%
'''
Fig 2F, Top 10 PC's best correlation with all 5 stim maps.

'''
real_best_corr = all_pc_best_corr.flatten()
shuffle_best_corr = all_pc_best_corr_s.flatten()
plotable = pd.DataFrame(0.0,columns = ['Best Corr','Data Type'],index = range(len(real_best_corr)+len(shuffle_best_corr)))
plotable.iloc[:len(real_best_corr),0] = real_best_corr
plotable.iloc[:len(real_best_corr),1] = 'Real PC'
plotable.iloc[len(real_best_corr):,0] = shuffle_best_corr
plotable.iloc[len(real_best_corr):,1] = 'Shuffled PC'

#%%
plt.clf()
plt.cla()
fontsize = 12

fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (4,6),dpi = 300)
sns.histplot(data = plotable,x = 'Best Corr',ax = ax,hue = 'Data Type', stat="percent",common_norm=False,bins = np.linspace(0,0.8,15),edgecolor='none',alpha = 0.8,hue_order = ['Shuffled PC','Real PC'])
# bins = [0,0.05,0.1,0.15,0.2,0.4,0.6,0.8]
# ax.set_xlabel('PC',size = 12)
# ax.set_ylabel('Explained Ratio(%)',size = 12)
# ax.set_title('Each PC explained Variance',size = 14)

ax.legend(['Real','Shuffled'],prop = { "size": fontsize })
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_yticks([0,20,40,60])
ax.set_yticklabels([0,20,40,60],fontsize = fontsize)

ax.set_xticks(np.arange(0,1,0.2))
ax.set_xticklabels([0,0.2,0.4,0.6,0.8],fontsize = fontsize)