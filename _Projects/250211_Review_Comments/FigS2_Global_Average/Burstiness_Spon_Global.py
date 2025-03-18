'''
This will show global ensemble's burstiness. This shall be almost the same as that of cellular response.

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
all_path_dic_v2 = list(ot.Get_Subfolders(r'D:\_DataTemp\_Fig_Datas\_All_Spon_Data_V2'))


#%% get event train of all locs.
burstiness = pd.DataFrame(columns = ['Loc','Burstiness'])
burstiness_rand = pd.DataFrame(columns = ['Loc','Burstiness'])
waittime_v1 = pd.DataFrame(index = range(100000),columns = ['Loc','Waittime'])


thres = 1
N_shuffle = 10
counter=0

for i,cloc in tqdm(enumerate(all_path_dic)):
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    cloc_name = cloc.split('\\')[-1]
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    start = c_spon.index[0]
    end = c_spon.index[-1]
    c_spon = Z_refilter(ac,'1-001',start,end).T
    c_on_frame = c_spon>thres
    c_ensemble = np.array(c_on_frame.mean(1))
    peaks,_ = find_peaks(c_ensemble,height = 0.1,distance = 5)
    c_raster = np.zeros(len(c_ensemble))
    c_raster[peaks]=1
    waittimes = np.diff(np.where(c_raster==1)[0])
    for k,c_waittime in enumerate(waittimes):
            waittime_v1.loc[counter] = [cloc_name,c_waittime]
            counter+=1
    cc_bur = Burstiness_Index_JN(c_raster,winnum=160)
    for k in range(N_shuffle):
        c_raster_s = Rand_Series(int(c_raster.sum()),len(c_raster))
        cc_bur_s = Burstiness_Index_JN(c_raster_s,winnum=160)
        burstiness_rand.loc[len(burstiness_rand)] = [cloc_name,cc_bur_s]
    burstiness.loc[len(burstiness)] = [cloc_name,cc_bur]

ot.Save_Variable(savepath,'Burstiness_V1_global',burstiness)
ot.Save_Variable(savepath,'Burstiness_V1_shuffle10_global',burstiness_rand)
waittime_v1 = waittime_v1.dropna(how='any')
ot.Save_Variable(savepath,'Waittime_V1_global',waittime_v1)

#%% Do the same on V2
burstiness_v2 = pd.DataFrame(columns = ['Loc','Burstiness'])
burstiness_rand_v2 = pd.DataFrame(columns = ['Loc','Burstiness'])
waittime_v2 = pd.DataFrame(index = range(100000),columns = ['Loc','Waittime'])


thres = 1
N_shuffle = 10
counter=0

for i,cloc in tqdm(enumerate(all_path_dic_v2)):
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    cloc_name = cloc.split('\\')[-1]
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    start = c_spon.index[0]
    end = c_spon.index[-1]
    c_spon = Z_refilter(ac,'1-001',start,end).T
    c_on_frame = c_spon>thres
    c_ensemble = np.array(c_on_frame.mean(1))
    peaks,_ = find_peaks(c_ensemble,height = 0.1,distance = 5)
    c_raster = np.zeros(len(c_ensemble))
    c_raster[peaks]=1
    waittimes = np.diff(np.where(c_raster==1)[0])
    for k,c_waittime in enumerate(waittimes):
            waittime_v1.loc[counter] = [cloc_name,c_waittime]
            counter+=1
    cc_bur = Burstiness_Index_JN(c_raster,winnum=160)
    for k in range(N_shuffle):
        c_raster_s = Rand_Series(int(c_raster.sum()),len(c_raster))
        cc_bur_s = Burstiness_Index_JN(c_raster_s,winnum=160)
        burstiness_rand_v2.loc[len(burstiness_rand_v2)] = [cloc_name,cc_bur_s]
    burstiness_v2.loc[len(burstiness_v2)] = [cloc_name,cc_bur]

ot.Save_Variable(savepath,'Burstiness_V2_global',burstiness_v2)
ot.Save_Variable(savepath,'Burstiness_V2_shuffle10_global',burstiness_rand_v2)
waittime_v2 = waittime_v2.dropna(how='any')
ot.Save_Variable(savepath,'Waittime_V2_global',waittime_v2)

#%% Plot scatterplot of global ensemble, the same style as that in cellular.
burstiness['Area']='V1'
burstiness_v2['Area']='V2'
burstiness_rand['Area']='V1'
burstiness_rand_v2['Area']='V2'
burstiness_comb = pd.concat([burstiness,burstiness_v2])
burstiness_comb_s = pd.concat([burstiness_rand,burstiness_rand_v2])
burstiness_comb['Data_Type'] = 'Real'
burstiness_comb_s['Data_Type'] = 'Shuffle'
#%% plot part
plotable = pd.concat([burstiness_comb,burstiness_comb_s])
all_locs = list(set(plotable['Loc']))
mean_scatter = pd.DataFrame(columns=['Loc','Burstiness','Area','Data_Type'])
for i,cloc in enumerate(all_locs):
    c_real = plotable.groupby(['Loc','Data_Type']).get_group((cloc,'Real'))
    c_shuffle = plotable.groupby(['Loc','Data_Type']).get_group((cloc,'Shuffle'))
    c_area = c_real.iloc[0,-2]
    c_real_bi = c_real['Burstiness'].mean()
    c_shuffle_bi = c_shuffle['Burstiness'].mean()
    mean_scatter.loc[len(mean_scatter)] = [cloc,c_real_bi,c_area,'Real']
    mean_scatter.loc[len(mean_scatter)] = [cloc,c_shuffle_bi,c_area,'Shuffle']
    

fontsize = 14
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(3.5,5),dpi=300)

sns.stripplot(data=mean_scatter,x = 'Area',y='Burstiness',hue='Data_Type',color='black',s=3,ax=ax, dodge=True,legend=False,jitter=True,alpha=0.7)
sns.barplot(data = plotable,x = 'Area',y='Burstiness',hue='Data_Type',width=0.75,capsize=0.15,ax = ax,legend=False,lw=1)

ax.set_ylim(0,0.15)
ax.set_yticks([0,0.05,0.1,0.15])
ax.set_yticklabels([0,0.05,0.1,0.15],fontsize = fontsize)
ax.set_xticklabels(['V1','V2'],fontsize = fontsize)

ax.set_ylabel('')
ax.set_xlabel('')
fig.savefig(ot.join(savepath,'Burstiness_Compare_Global.png'),bbox_inches = 'tight')
