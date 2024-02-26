'''
Evaluate data quality. just show cell infos.

'''
#%%

import OS_Tools_Kit as ot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

wp = r'D:\ZR\_Data_Temp\2pt_T151425_A2\_CAIMAN'
all_dics = ot.Load_Variable(wp,'All_Comp_Trains.pkl')
real_cell_dic = ot.Load_Variable(wp,'Component_ID_Lists.pkl')

#%%
cc_mask = np.zeros(shape = (512,512),dtype = 'f8')
real_cell_num = len(real_cell_dic.keys())
frame_num = len(all_dics[1]['1-001'])
all_cell_frame = pd.DataFrame(columns = list(range(real_cell_num)),index = list(range(frame_num)))
for i in tqdm(range(real_cell_num)):
    cc = real_cell_dic[i+1]
    cc_train = all_dics[cc+1]['1-001']
    all_cell_frame[i] = cc_train
    cc_mask += all_dics[cc+1]['Cell_Mask']
ot.Save_Variable(wp,'Accepted_Cell_Frame',all_cell_frame)
#%% Draw all comp map.
import Graph_Operation_Kit as gt
all_mask = np.zeros(shape = (512,512),dtype = 'f8')
for i in tqdm(range(1104)):
    x,y = all_dics[i+1]['Cell_Loc']
    if x>20 and x<492:
        if y>20 and y<492:
            all_mask += all_dics[i+1]['Cell_Mask']
clipped_all_mask = gt.Clip_And_Normalize(all_mask,clip_std=5)
gt.Show_Graph(clipped_all_mask,'All_Comps',wp)
#%%
a = all_cell_frame.iloc[:,67]
plt.switch_backend('webAgg')
fig,ax = plt.subplots(1,figsize = (10,6))
ax = plt.plot(a)
plt.show()
#%% Get dF/F values.
from Filters import Signal_Filter
passed_band = (0.05,3)
fps = 31
filted_frame = np.zeros(shape = (259,60000),dtype = 'f8')
dff_frame = np.zeros(shape = (259,60000),dtype = 'f8')
for i in range(259):
    c_series = all_cell_frame.iloc[:60000,i]
    filted_series = Signal_Filter(c_series,filter_para = (passed_band[0]*2/fps,passed_band[1]*2/fps))
    filted_frame[i,:] =  filted_series
    dff_frame[i,:] = (filted_series-filted_series.mean())/filted_series.mean()
#%% bin first
bin4_frame = all_cell_frame.groupby(np.arange(len(all_cell_frame))//4).mean()
passed_band = (0.05,3)
fps = 31/4
filted_frame = np.zeros(shape = (259,15000),dtype = 'f8')
dff_frame = np.zeros(shape = (259,15000),dtype = 'f8')
for i in range(259):
    c_series = bin4_frame.iloc[:15000,i]
    filted_series = Signal_Filter(c_series,filter_para = (passed_band[0]*2/fps,passed_band[1]*2/fps))
    filted_frame[i,:] =  filted_series
    c_dff_series = (filted_series-filted_series.mean())/(filted_series.mean())
    dff_frame[i,:] = c_dff_series/c_dff_series.std()
#%% Do PCA analysis
from Series_Analyzer.Cell_Frame_PCA_Cai import Do_PCA
import cv2
dff_array = pd.DataFrame(dff_frame)
comp,info,weight = Do_PCA(dff_array)
#%% visualize all PC
import seaborn as sns
save_folder = wp+r'\_PCA_Results'
ot.mkdir(save_folder)
for j in tqdm(range(259)):
    c_comp = comp.iloc[:,j]
    c_PC_graph = np.zeros(shape = (512,512),dtype = 'f8')
    for i in range(259):# cv2 will load frame in sequence x,y.
        cc = real_cell_dic[i+1]
        cc_loc = (all_dics[cc+1]['Cell_Loc'].astype('i4')[1],all_dics[cc+1]['Cell_Loc'].astype('i4')[0])
        c_resp = c_comp.loc[i]
        c_PC_graph = cv2.circle(c_PC_graph,cc_loc,4,c_resp,-1)
    fig = plt.figure(figsize = (15,15))
    plt.title('PC'+str(j+1),fontsize=36)
    fig = sns.heatmap(c_PC_graph,square=True,yticklabels=False,xticklabels=False,center = 0)
    fig.figure.savefig(ot.join(save_folder,str(j+1))+'.png')
    plt.clf()
    plt.close()
