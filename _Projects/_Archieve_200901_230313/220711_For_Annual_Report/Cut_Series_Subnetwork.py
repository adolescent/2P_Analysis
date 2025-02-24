# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 13:08:30 2022

@author: ZR
"""



from Series_Analyzer.Preprocessor_Cai import Pre_Processor_Cai
import OS_Tools_Kit as ot
from Series_Analyzer.Pairwise_Correlation import Series_Cut_Pair_Corr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr,spearmanr
import statsmodels.api as sm
import pandas as pd
from tqdm import tqdm
from Series_Analyzer.Series_Cutter import Series_Window_Slide
wp = r'D:\ZR\_Temp_Data\220711_temp'

#%% Initailization

pcinfo91 = ot.Load_Variable(wp,'pc91_info.pkl')
pcwin91 = ot.Load_Variable(wp,'pc91win.pkl')

pc_tuned = pcinfo91[pcinfo91.OD_A != -999]
pc_tuned = pc_tuned[pc_tuned.OD_B != -999]
pc_tuned = pc_tuned[pc_tuned.Orien_A != -999]
pc_tuned = pc_tuned[pc_tuned.Orien_B != -999]
tuned_lists = pc_tuned.index
pcinfo91_used = pcinfo91.loc[tuned_lists,:]
OD_diff = abs(pcinfo91_used['OD_A']-pcinfo91_used['OD_B'])
raw_orien_diff = abs(pcinfo91_used['Orien_A']-pcinfo91_used['Orien_B'])
raw_orien_diff[raw_orien_diff>90]=(180-raw_orien_diff)
Orien_diff = raw_orien_diff
pcinfo91_used['OD_diff'] = OD_diff
pcinfo91_used['Orien_diff'] = Orien_diff
ot.Save_Variable(wp, 'pc91tuned_info', pcinfo91_used)
pcwin91_tuned = pcwin91.loc[tuned_lists,:]
ot.Save_Variable(wp, 'pc91win_tuned', pcwin91_tuned)

#%% get subnetworks.
pcinfo91t = ot.Load_Variable(wp,'pc85tuned_info.pkl')
pcwin91t = ot.Load_Variable(wp,'pc85win_tuned.pkl')
avr91 = pcwin91t.mean()
# similar eyes
RE_pairs = pcinfo91t[pcinfo91t['OD_A']<-0.5][pcinfo91t['OD_B']<-0.5]
LE_pairs = pcinfo91t[pcinfo91t['OD_A']>0.5][pcinfo91t['OD_B']>0.5]
sameeye_pairs = pd.concat([RE_pairs,LE_pairs])
sameeye_index = sameeye_pairs.index
sameeye_win = pcwin91t.loc[sameeye_index,:]
avr_sameeye = sameeye_win.mean()
# similar orientation
sameorien_pairs = pcinfo91t[pcinfo91t['Orien_diff']<22.5]
sameorien_index = sameorien_pairs.index
sameorien_win = pcwin91t.loc[sameorien_index,:]
avr_sameorien = sameorien_win.mean()

plt.plot(avr_sameeye[60:])
plt.plot(avr_sameorien[60:])
plt.plot(avr91[60:])

plt.plot((avr_sameeye-avr91)[60:])
plt.plot((avr_sameorien-avr91)[60:])
plt.plot(avr91)

# need shuffle to see it's not random results.
time = 2000
A_matrix = np.zeros(shape = (2000,187))
B_matrix = np.zeros(shape = (2000,187))
r_lists = []

for i in tqdm(range(time)):
    samp1 = pcwin91t.sample(20000).mean()-avr91
    samp2 = pcwin91t.sample(20000).mean()-avr91
    A_matrix[i,:] = samp1
    B_matrix[i,:] = samp2
    cr,_ = spearmanr(samp1[30:140],samp2[30:140])
    r_lists.append(cr)
    

#%% Then seperate orien,od network here.
from Series_Analyzer.Pair_Corr_Tools import Win_Corr_Select
LE_index = LE_pairs.index
RE_index = RE_pairs.index
LE_win = pcwin91t.loc[LE_index,:]
RE_win = pcwin91t.loc[RE_index,:]
LR_info = Win_Corr_Select((pcinfo91t['OD_A']>0.5)*(pcinfo91t['OD_B']<-0.5), pcwin91t)
LR_info2 = Win_Corr_Select((pcinfo91t['OD_A']<-0.5)*(pcinfo91t['OD_B']>0.5), pcwin91t)
LR_info = pd.concat([LR_info,LR_info2]).index
LR_win = pcwin91t.loc[LR_info,:]
# Seperate orientation diff into 4 group.
pcinfo91t['Orien_group'] = pcinfo91t['Orien_diff']//22.5+1
#orien_groups = dict(list(pcinfo91t.groupby('Orien_group'))) #Use function.
orien1_win = Win_Corr_Select(pcinfo91t['Orien_group']==1, pcwin91t)
orien2_win = Win_Corr_Select(pcinfo91t['Orien_group']==2, pcwin91t)
orien3_win = Win_Corr_Select(pcinfo91t['Orien_group']==3, pcwin91t)
orien4_win = Win_Corr_Select(pcinfo91t['Orien_group']==4, pcwin91t)
# get similar orientation subgroups.
pcinfo91t['Orien_A_group'] = pcinfo91t['Orien_A']//45+1
pcinfo91t['Orien_B_group'] = pcinfo91t['Orien_B']//45+1
orien11_win = Win_Corr_Select((pcinfo91t['Orien_A_group']==1)*(pcinfo91t['Orien_B_group']==1)*(pcinfo91t['Orien_group']==1),pcwin91t)
orien22_win = Win_Corr_Select((pcinfo91t['Orien_A_group']==2)*(pcinfo91t['Orien_B_group']==2)*(pcinfo91t['Orien_group']==1),pcwin91t)
orien33_win = Win_Corr_Select((pcinfo91t['Orien_A_group']==3)*(pcinfo91t['Orien_B_group']==3)*(pcinfo91t['Orien_group']==1),pcwin91t)
orien44_win = Win_Corr_Select((pcinfo91t['Orien_A_group']==4)*(pcinfo91t['Orien_B_group']==4)*(pcinfo91t['Orien_group']==1),pcwin91t)



plt.plot(orien11_win.mean())
plt.plot(orien22_win.mean())
plt.plot(orien33_win.mean())
plt.plot(orien44_win.mean())
#plt.plot(orien1_win.mean())
plt.plot(avr91)

plt.plot(orien11_win.mean()[60:])
plt.plot(orien22_win.mean()[60:])
plt.plot(orien33_win.mean()[60:])
plt.plot(orien44_win.mean()[60:])
plt.plot(avr91[60:])

plt.plot(LE_win.mean()[60:])
plt.plot(RE_win.mean()[60:])
plt.plot(LR_win.mean()[60:])
plt.plot(avr91[60:])

plt.plot((LE_win.mean()-avr91)[60:])
plt.plot((RE_win.mean()-avr91)[60:])
plt.plot((LR_win.mean()-avr91)[60:])
#plt.plot((A_matrix.mean(0))[60:])


plt.plot(orien1_win.mean()[60:])
plt.plot(orien2_win.mean()[60:])
plt.plot(orien3_win.mean()[60:])
plt.plot(orien4_win.mean()[60:])
plt.plot(avr91[60:])


plt.plot((orien11_win.mean()-avr91))
plt.plot((orien22_win.mean()-avr91))
plt.plot((orien33_win.mean()-avr91))
plt.plot((orien44_win.mean()-avr91))
plt.plot((LE_win.mean()-avr91)[60:])
plt.plot((RE_win.mean()-avr91)[60:])
plt.plot((LR_win.mean()-avr91)[60:])
#%% Compare with Cell response, sort cell sequence by tuning.
acd = ot.Load_Variable(wp,r'Series_91_Run1.pkl')
sns.heatmap(acd,center = 0)
# sort them by pref-corr and pref-OD
all_cell_tuning_91 = ot.Load_Variable(r'D:\ZR\_Temp_Data\220420_L91\_CAIMAN\Cell_Tuning_Dic.pkl')
acn = list(all_cell_tuning_91.keys())
tuning_frame = pd.DataFrame(columns = ['OD','Orien'])
for i,cc in enumerate(acn):
    c_od = all_cell_tuning_91[cc]['OD']['Tuning_Index']
    c_orien = all_cell_tuning_91[cc]['Fitted_Orien']
    if c_orien != 'No_Tuning':
        tuning_frame.loc[cc] = [c_od,c_orien]
OD_sorted = tuning_frame.sort_values('OD')
Orien_sorted = tuning_frame.sort_values('Orien')
od_sorted_cell = acd.reindex(index = OD_sorted.index)
sns.heatmap(od_sorted_cell,center = 0)
# regress global avr
global_avr = od_sorted_cell.mean(0)
a = od_sorted_cell-global_avr
sns.heatmap(a,center = 0,vmax = 2,vmin = -2)
# orien map.
orien_sorted_cell = acd.reindex(index = Orien_sorted.index)
sns.heatmap(orien_sorted_cell,center = 0)
# sort pairwise corr and avr by group.
global_avr = orien_sorted_cell.mean(0)
a = orien_sorted_cell-global_avr
sns.heatmap(a,center = 0)
# firing counter in time window, let's see whether they are related.
thres_od_sorted_cell =orien_sorted_cell[orien_sorted_cell>3]
thres_od_sorted_cell = thres_od_sorted_cell.fillna(0)
#thres_od_sorted_cell = od_sorted_cell>3
cutted_od_sorted_cell = Series_Window_Slide(thres_od_sorted_cell)
a = cutted_od_sorted_cell.sum(1)
sns.heatmap(a,center = 0,vmax = 150)
#%% Resort pairwise-corr to see how inner relationship effect corr.
pc91_tuned = ot.Load_Variable(wp,'pc91tuned_info.pkl')
pc_od_sort = pcwin91.reindex(pc91_tuned.sort_values('OD_diff').index)
pc_orien_sort = pcwin91.reindex(pc91_tuned.sort_values('Orien_diff').index)
pc_dist_sort = pcwin91.reindex(pc91_tuned.sort_values('Dist').index)

a = pc_od_sort.groupby(np.arange(len(pc_od_sort))//355).mean()
#pc_rand = pcwin91.sample(177310)
#a = pc_rand.groupby(np.arange(len(pc_rand))//355).mean()
c_info = pc91_tuned.sort_values('OD_diff').groupby(np.arange(len(pc91_tuned.sort_values('OD_diff')))//355).mean()

num_ticks = 50
# the index of the position of yticks
yticks = np.linspace(0, 500-1, num_ticks, dtype=np.int)
# the content of labels of these yticks
yticklabels = [c_info['OD_diff'].round(3)[idx] for idx in yticks]
ax = sns.heatmap(a)
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_title('OD diff vs Correlation')
plt.show()
# mean subtraction
b = a-a.mean(0)
ax = sns.heatmap(b,center = 0,vmax = 0.1,vmin = -0.1)
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_title('OD diff vs Correlation')
plt.show()

#%% Do ther same thing on L76 & L85 data.
pcinfo85 = ot.Load_Variable(wp,'pc85tuned_info.pkl')
pcwin85 = ot.Load_Variable(wp,'pc85win_tuned.pkl')

pc_od_sort = pcwin85.reindex(pcinfo85.sort_values('OD_diff').index)
pc_orien_sort = pcwin85.reindex(pcinfo85.sort_values('Orien_diff').index)
pc_dist_sort = pcwin85.reindex(pcinfo85.sort_values('Dist').index)

a = pc_od_sort.groupby(np.arange(len(pc_od_sort))//324).mean()
#pc_rand = pcwin91.sample(177310)
#a = pc_rand.groupby(np.arange(len(pc_rand))//325).mean()
c_info = pcinfo85.sort_values('OD_diff').groupby(np.arange(len(pcinfo85.sort_values('OD_diff')))//324).mean()

num_ticks = 50
# the index of the position of yticks
yticks = np.linspace(0, 500-1, num_ticks, dtype=np.int)
# the content of labels of these yticks
yticklabels = [c_info['OD_diff'].round(3)[idx] for idx in yticks]
ax = sns.heatmap(a)
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_title('OD diff vs Correlation')
plt.show()
# mean subtraction
b = a-a.mean(0)
ax = sns.heatmap(b,center = 0,vmax = 0.075,vmin = -0.075)
#ax = sns.heatmap(b,center = 0)
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_title('OD diff vs Correlation')
plt.show()

##############L76###############
pcinfo76 = ot.Load_Variable(wp,'pc76tuned_info.pkl')
pcwin76 = ot.Load_Variable(wp,'pc76win_tuned.pkl')

pc_od_sort = pcwin76.reindex(pcinfo76.sort_values('OD_diff').index)
pc_orien_sort = pcwin76.reindex(pcinfo76.sort_values('Orien_diff').index)
pc_dist_sort = pcwin76.reindex(pcinfo76.sort_values('Dist').index)

a = pc_orien_sort.groupby(np.arange(len(pc_orien_sort))//174).mean()
#pc_rand = pcwin91.sample(177310)
#a = pc_rand.groupby(np.arange(len(pc_rand))//325).mean()
c_info = pcinfo76.sort_values('Orien_diff').groupby(np.arange(len(pcinfo76.sort_values('Orien_diff')))//174).mean()

num_ticks = 50
# the index of the position of yticks
yticks = np.linspace(0, 500-1, num_ticks, dtype=np.int)
# the content of labels of these yticks
yticklabels = [c_info['Orien_diff'].round(3)[idx] for idx in yticks]
ax = sns.heatmap(a.iloc[:,60:])
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_title('Orien diff vs Correlation')
plt.show()
# mean subtraction
b = a-a.mean(0)
ax = sns.heatmap(b.iloc[:,60:],center = 0,vmax = 0.1,vmin = -0.1)
#ax = sns.heatmap(b.iloc[:,60:],center = 0)
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_title('Orien diff vs Correlation')
plt.show()


