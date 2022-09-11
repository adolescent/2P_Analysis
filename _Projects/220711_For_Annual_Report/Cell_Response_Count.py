# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 12:31:37 2022

@author: ZR

This script count cell response of every single frame. Counting single cell spon resposne.

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
import cv2

wp = r'F:\_Data_Temp\220711_temp'

#%% Initailization
acd = ot.Load_Variable(wp,'All_Series_Dic91.pkl')
acinfo = ot.Load_Variable(wp,'Cell_Tuning_Dic91.pkl')
dataframe1 = ot.Load_Variable(wp,'Series_91_Run1.pkl')
spikes = dataframe1[dataframe1>2]
spikes = spikes.fillna(0).clip(upper = 5)
sns.heatmap(spikes,center = 0,vmax = 5)
#%% get tuning frames.
actune = pd.DataFrame(columns = ['OD','Orien'])
acn = list(acd.keys())
for i,cc in enumerate(acn):
    tc = acinfo[cc]
    if tc['Fitted_Orien'] != 'No_Tuning':
        actune.loc[cc] = [tc['OD']['Tuning_Index'],tc['Fitted_Orien']]
        
od_sorted_index = actune.sort_values('OD').index
orien_sorted_index = actune.sort_values('Orien').index
od_sorted_frame = spikes.reindex(od_sorted_index)
orien_sorted_frame = spikes.reindex(orien_sorted_index)
sns.heatmap(od_sorted_frame,center = 0)
# average every 50 lines.
#a = od_sorted_frame.groupby(np.arange(len(od_sorted_frame))//50).mean()
# visualize spike change.
tuned_spikes = spikes.reindex(actune.index)
cellnum,framenum = tuned_spikes.shape
tuned_cell = list(tuned_spikes.index)
ot.Save_Variable(wp, 'Tuned_Spikes_91', tuned_spikes)
#%% Video generation
all_run01_frame = np.zeros(shape = (512,512,framenum))
for i,cc in tqdm(enumerate(tuned_cell)):
    ccy,ccx = acd[cc]['Cell_Loc']
    ccy = int(ccy)
    ccx = int(ccx)
    for j in range(framenum):
        c_strengh = float(tuned_spikes.loc[cc,j])
        if c_strengh !=0:
            all_run01_frame[:,:,j] = cv2.circle(img = np.float32(all_run01_frame[:,:,j]),center = (ccx,ccy),radius = 5,color = c_strengh,thickness = -1)
#ot.Save_Variable(wp, 'L91_Spike_Video_Run01', all_run01_frame)
# Write video.
data_for_write = (all_run01_frame*51).astype('u1')
fps = 8
graph_size = (512,512)
video_writer = cv2.VideoWriter(wp+r'\\L85_Run01_Video.mp4',cv2.VideoWriter_fourcc('X','V','I','D'),fps,graph_size,0)
for i in range(framenum):
    u1_writable_graph = data_for_write[:,:,i]
    annotated_graph = cv2.putText(u1_writable_graph.astype('f4'),'Stim ID = '+str(i),(250,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,255,1)
    video_writer.write(annotated_graph.astype('u1'))
del video_writer 
#%% Calculate each frame sumed network response.
# 1. No tuning weight ver.
spike_info_column = ['All_Num','All_spike','LE_Num','LE_spike','RE_Num','RE_spike','Orien0_Num','Orien0_spike','Orien45_Num','Orien45_spike','Orien90_Num','Orien90_spike','Orien135_Num','Orien135_spike']
# Use 0,45,90,135+-22.5 as group order.
actune['Orien_Group'] = (((actune['Orien']+22.5)%180)//45)+1
cell_response_frame = pd.DataFrame(0,columns=spike_info_column,index = range(framenum))
OD_thres = 0.5
LE_cell_Num = (actune['OD']>OD_thres).sum()
RE_cell_Num = (actune['OD']<-OD_thres).sum()
Orien0_cell_Num = (actune['Orien_Group']==1).sum()
Orien45_cell_Num = (actune['Orien_Group']==2).sum()
Orien90_cell_Num = (actune['Orien_Group']==3).sum()
Orien135_cell_Num = (actune['Orien_Group']==4).sum()

for i in tqdm(range(framenum)):
    #cframe = tuned_spikes.loc[:,i]
    cframe = tuned_spikes.loc[:,i]
    firing_cells = cframe[cframe>0]
    cell_response_frame.loc[i,'All_Num'] = len(firing_cells)
    cell_response_frame.loc[i,'All_spike'] = firing_cells.sum()
    fire_cell_names = list(firing_cells.index)
    # then cycle 
    for j,cc in enumerate(fire_cell_names):
        c_tune = actune.loc[cc,:]
        if c_tune['OD']>OD_thres: # LE cell
            cell_response_frame.loc[i,'LE_Num'] +=1
            cell_response_frame.loc[i,'LE_spike'] += firing_cells[cc]
        elif c_tune['OD']<-OD_thres: # RE cell
            cell_response_frame.loc[i,'RE_Num'] +=1
            cell_response_frame.loc[i,'RE_spike'] += firing_cells[cc]
        # orien counter
        if c_tune['Orien_Group'] == 1 : # Orien 0 cell
            cell_response_frame.loc[i,'Orien0_Num'] +=1
            cell_response_frame.loc[i,'Orien0_spike'] += firing_cells[cc]
        elif c_tune['Orien_Group'] == 2 : # Orien 45 cell
            cell_response_frame.loc[i,'Orien45_Num'] +=1
            cell_response_frame.loc[i,'Orien45_spike'] += firing_cells[cc]
        elif c_tune['Orien_Group'] == 3 : # Orien 90 cell
            cell_response_frame.loc[i,'Orien90_Num'] +=1
            cell_response_frame.loc[i,'Orien90_spike'] += firing_cells[cc]
        elif c_tune['Orien_Group'] == 4 : # Orien 135 cell
            cell_response_frame.loc[i,'Orien135_Num'] +=1
            cell_response_frame.loc[i,'Orien135_spike'] += firing_cells[cc]
    
ot.Save_Variable(wp, 'Network_activity_91',cell_response_frame)
# find peak
from scipy.signal import find_peaks
x = cell_response_frame['All_Num']
peaks, properties = find_peaks(x, height=10,distance = 3,width = 0)
plt.plot(x)
plt.plot(np.zeros_like(x), "--", color="gray")
#plt.plot(cell_response_frame['All_Num'])
#plt.plot(cell_response_frame['LE_Num']/89)
#plt.plot(cell_response_frame['RE_Num']/135)
plt.plot(peaks, x[peaks], "x")
plt.show()

# =============================================================================
# # This part show peak num vs least cell size. 
# a = []
# for i in range(1,50):
#     peaks, _ = find_peaks(x, height=i/596,distance = 3)
#     a.append(len(peaks))
# plt.plot(a)
# =============================================================================
#%% Use spike cell num to define active events.
peak_info = cell_response_frame.loc[peaks,:]
peak_info['LE_ON'] = peak_info['LE_Num']>5
peak_info['RE_ON'] = peak_info['RE_Num']>5
LE_ON = (peak_info[peak_info['LE_ON']==True]).index
RE_ON = (peak_info[peak_info['RE_ON']==True]).index
plt.plot(x)
plt.plot(peaks, x[peaks], "o",color = 'gray')# all peak
plt.plot(RE_ON, x[RE_ON], "+",color = 'red')# all peak
plt.plot(LE_ON, x[LE_ON], "x",color = 'yellow')# all peak
plt.show()


#%% determine how event thres affect event number.
peak_info = cell_response_frame.loc[peaks,:]
thres = np.arange(300,0,-2)
thres_frame = pd.DataFrame(0,columns = ['thres','LE_all','RE_all','Coactive','LE_alone','RE_alone'],index = range(len(thres)))
for i,c_thres in enumerate(thres):
    peak_info['LE_ON'] = peak_info['LE_Num']>c_thres
    peak_info['RE_ON'] = peak_info['RE_Num']>c_thres
    c_LE = peak_info['LE_ON'].sum()
    c_RE = peak_info['RE_ON'].sum()
    c_coact = ((peak_info['RE_ON']==True)*(peak_info['LE_ON']==True)).sum()
    c_LE_alone = c_LE-c_coact
    c_RE_alone = c_RE-c_coact
    thres_frame.loc[i,:] = [c_thres,c_LE,c_RE,c_coact,c_LE_alone,c_RE_alone]
    
used_frame = thres_frame.melt(id_vars = ['thres'], var_name='Type',value_name = 'Count')

fig, ax = plt.subplots()
sns.lineplot(data = used_frame,x = 'thres',y = 'Count',hue = 'Type',ax = ax)  # distplot is deprecate and replaced by histplot
ax.set_xlim(150,0)
plt.show()
#%% determine thres of BIG and SMALL response,
peak_info = cell_response_frame.loc[peaks,:]
peak_info['LE_ON'] = peak_info['LE_Num']>5
peak_info['RE_ON'] = peak_info['RE_Num']>5
peak_info['Single_ON'] = (peak_info['LE_ON']!=peak_info['RE_ON'])
peak_info['Both_ON'] = (peak_info['LE_ON']*peak_info['RE_ON'])
peak_info['Eye_ON'] = (peak_info['LE_ON']+peak_info['RE_ON'])# at least one eye network on.
ot.Save_Variable(wp, 'peak_info_91', peak_info)
# count big/small number and single/both propotion.
# =============================================================================
# big_thres = 35 # define which peak is big and which is small.
# peak_info['Big'] = peak_info['All_Num']>big_thres
# big_num = peak_info['Big'].sum()
# small_num = (peak_info['Big']==False).sum()
# big_both_prop = (peak_info[peak_info['Big']==True])['Both_ON'].sum()/big_num
# big_single_prop = (peak_info[peak_info['Big']==True])['Single_ON'].sum()/big_num
# small_both_prop = (peak_info[peak_info['Big']==False])['Both_ON'].sum()/small_num
# small_single_prop = (peak_info[peak_info['Big']==False])['Single_ON'].sum()/small_num
# =============================================================================
# cycle network smaller than specific, and calculate single/both prop.

Peak_Prop = pd.DataFrame(0,columns = ['Scale','Peak_Num','Peak_Num_Eye','Single_prop','Both_prop','Single_prop_eye','Both_prop_eye'],index = range(11,150))
for i in tqdm(range(11,150)):
    c_net = peak_info[(peak_info['All_Num']>i)*(peak_info['All_Num']<i+20)]
    c_peak_num = len(c_net)
    c_Both_num = c_net['Both_ON'].sum()
    c_Single_num = c_net['Single_ON'].sum()
    c_Eye_num = c_net['Eye_ON'].sum()
    Peak_Prop.loc[i,:] = [i,c_peak_num,c_Eye_num,c_Single_num/c_peak_num,c_Both_num/c_peak_num,c_Single_num/c_Eye_num,c_Both_num/c_Eye_num]
  
    
plt.plot(Peak_Prop['Single_prop'])
plt.plot(Peak_Prop['Both_prop'])
plt.plot(Peak_Prop['Both_prop']+Peak_Prop['Single_prop'])

plt.plot(Peak_Prop['Single_prop_eye'])
plt.plot(Peak_Prop['Both_prop_eye'])

#%% return maps.
# first,return all map avr.
LE_peaks = peak_info[(peak_info['RE_ON']==True)*(peak_info['LE_ON']==False)].sort_values('RE_spike',ascending = False)
#LE_peaks = peak_info[(peak_info['RE_ON']==True)].sort_values('RE_spike',ascending = False)

LE_best_frames = LE_peaks.index[:100]
LE_restore = tuned_spikes.loc[:,LE_best_frames].mean(1)
LE_restore_map = np.zeros(shape = (512,512))
for i,cc in enumerate(LE_restore.index):
    ccy,ccx = acd[cc]['Cell_Loc']
    ccy = int(ccy)
    ccx = int(ccx)
    LE_restore_map = cv2.circle(img = np.float32(LE_restore_map),center = (ccx,ccy),radius = 5,color = LE_restore[cc],thickness = -1)
    
sns.heatmap(LE_restore_map,center = 0,square = True,xticklabels=False,yticklabels=False)
# count network scale of single and both networks.
single_peaks = peak_info[peak_info['Single_ON']==True]['All_Num']
both_peaks = peak_info[peak_info['Both_ON']==True]['All_Num']

#%% Count cell tuning bais of each peak.
peak_info['LE_prop'] = peak_info['LE_spike']/LE_cell_Num
peak_info['RE_prop'] = peak_info['RE_spike']/RE_cell_Num
peak_info['Bias'] = abs(peak_info['LE_prop']-peak_info['RE_prop'])/(peak_info['LE_prop']+peak_info['RE_prop'])
peak_info['Scale'] = (peak_info['All_Num']//10)*10

sns.scatterplot(data = peak_info[peak_info['Eye_ON']==True],x = 'All_Num',y = 'Bias')
#sns.kdeplot(data = peak_info[peak_info['Eye_ON']==True],x = 'All_Num',y = 'Bias',fill=True, thresh=0, levels=100, cmap="mako")
sns.lmplot(data = peak_info,x = 'All_Num',y = 'Bias')


a = peak_info[peak_info['Eye_ON']==True].groupby('Scale').mean()
#a = peak_info.groupby('Scale').mean()
ax = plt.subplots()
sns.kdeplot(data = peak_info[peak_info['Eye_ON']==True],x = 'All_Num',y = 'Bias',fill=True, thresh=0, levels=100, cmap="mako")
sns.lineplot(data = a,x = 'Scale',y = 'Bias')
#%% count all network tuning. count all peak have at least one peak.
all_peak_name = peak_info.index
peak_bias = pd.DataFrame(columns = ['All_Num','Tune'])
for i,cp in enumerate(all_peak_name):
    cp_info = peak_info.loc[cp,:]
    if (cp_info['LE_Num']<5)*(cp_info['LE_Num']<5)*(cp_info['Orien0_Num']<5)*(cp_info['Orien45_Num']<5)*(cp_info['Orien90_Num']<5)*(cp_info['Orien135_Num']<5):
        continue
    c_LE_prop = cp_info['LE_Num']/LE_cell_Num
    c_RE_prop = cp_info['RE_Num']/RE_cell_Num
    c_Orien0_prop = cp_info['Orien0_Num']/Orien0_cell_Num
    c_Orien45_prop = cp_info['Orien45_Num']/Orien45_cell_Num
    c_Orien90_prop = cp_info['Orien90_Num']/Orien90_cell_Num
    c_Orien135_prop = cp_info['Orien135_Num']/Orien135_cell_Num
    c_group = np.array([c_LE_prop,c_RE_prop,c_Orien0_prop,c_Orien45_prop,c_Orien90_prop,c_Orien135_prop])
    c_mean = c_group.mean()
    c_max = c_group.max()
    c_tune = abs((c_max-c_mean)/c_mean)
    peak_bias.loc[i,:] = [cp_info['All_Num'],c_tune]
    
    
sns.scatterplot(data = peak_bias,x = 'All_Num',y = 'Tune',s = 2)
sns.kdeplot(data = peak_bias,x = 'All_Num',y = 'Tune',fill=True, thresh=0, levels=100, cmap="mako")

peak_bias['Scale'] = (peak_bias['All_Num']//10)*10
a = peak_bias.groupby('Scale').mean()
#a = peak_info.groupby('Scale').mean()
ax = plt.subplots()
sns.kdeplot(data = peak_bias,x = 'All_Num',y = 'Tune',fill=True, thresh=0, levels=100, cmap="mako")
sns.lineplot(data = a,x = 'Scale',y = 'Tune')
#%% simulate random signle network, will they make bigger network?
LE_num =(peak_info['LE_ON'] == True).sum()
RE_num =(peak_info['RE_ON'] == True).sum()
LE_rand = np.zeros(1040)
LE_rand[:LE_num]=1
RE_rand = np.zeros(1040)
RE_rand[:RE_num]=1
coact_num = []
for i in tqdm(range(100000)):
    np.random.seed(i)
    np.random.shuffle(LE_rand)
    np.random.shuffle(RE_rand)
    c_coact = (LE_rand*RE_rand).sum()
    coact_num.append(c_coact)

#plt.hist(coact_num,bins = 200)
plt.hist(coact_num, bins=range(int(min(coact_num)),int(max(coact_num)) + 2,2))
