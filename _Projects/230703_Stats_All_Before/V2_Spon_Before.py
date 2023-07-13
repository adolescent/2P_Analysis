'''
This script will generate V2 cell classes and do SVM seperator for V2 points.
'''

#%%
import OS_Tools_Kit as ot
import numpy as np
import pandas as pd
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
import Graph_Operation_Kit as gt
import cv2
import umap
import umap.plot
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from Kill_Cache import kill_all_cache
from sklearn.model_selection import cross_val_score
from sklearn import svm
from Cell_Tools.Cell_Visualization import Cell_Weight_Visualization
import random
import seaborn as sns
from My_Wheels.Cell_Class.Stim_Calculators import Stim_Cells
from My_Wheels.Cell_Class.Format_Cell import Cell
from Cell_Class.Advanced_Tools import *
from Cell_Class.Plot_Tools import *
from Stim_Frame_Align import One_Key_Stim_Align
from scipy.stats import pearsonr
import scipy.stats as stats
#%% 220221 L76 Loc 6
wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220211_L76_2P'
ac = Stim_Cells(day_folder = wp,od = False,od_type = False,orien =2,color = 5)
ac.Calculate_All()
ac.Save_Class()
ac.Plot_T_Graphs()
#%% 220914 L85
all_path = ot.Get_Sub_Folders(r'D:\ZR\_Data_Temp\All_V2_Cell_Classes')
cp = all_path[0]
ac = ot.Load_Variable(cp,'Cell_Class.pkl')
spon_frame = ac.Z_Frames['1-001']
spon_avr = spon_frame.mean(1)
reaction_cells = (spon_frame>2).sum(1)
plt.switch_backend('webAgg')
plt.plot(spon_avr)
# plt.plot(reaction_cells)
plt.show()
# Save 
# used_spon_frame = spon_frame.loc[6500:13200,:]
# ot.Save_Variable(cp,'Spon_Before',used_spon_frame)
#%% get umap and generate repeat 
all_path = ot.Get_Sub_Folders(r'D:\ZR\_Data_Temp\All_V2_Cell_Classes')
stats_info = pd.DataFrame(0,columns=['SVM_Correct','Spon_Frames','On_Frames','OD_repeats','Orien_repeats','Color_repeats','Spon_SVM_Train'],index=all_path)
for i,cp in tqdm(enumerate(all_path)):
    kill_all_cache(r'C:\ProgramData\anaconda3\envs\umapzr')
    ac = ot.Load_Variable(cp,'Cell_Class.pkl')
    spon_series = ot.Load_Variable(cp,'Spon_Before.pkl')
    spon_frame_num = spon_series.shape[0]
    all_stim_frame,all_stim_label = ac.Combine_Frame_Labels(od = False,orien = True,color = False,isi = True)
    # train an svm using umap embedded results.
    reducer = umap.UMAP(n_components=3,n_neighbors=20)
    reducer.fit(all_stim_frame)
    stim_embeddings = reducer.embedding_
    spon_embeddings = reducer.transform(spon_series)
    classifier,score = SVM_Classifier(embeddings=stim_embeddings,label = all_stim_label)
    svm_series = SVC_Fit(classifier,spon_embeddings,thres_prob = 0)
    on_num = np.sum(svm_series>0)
    od_num = np.sum((svm_series>0)*(svm_series<9))
    orien_num = np.sum((svm_series>8)*(svm_series<17))
    color_num = np.sum(svm_series>16)
    stats_info.loc[cp] = [score,spon_frame_num,on_num,od_num,orien_num,color_num,[svm_series]]
# Do some statistic on datas.
stats_info['On_freq'] = stats_info['On_Frames']*1.301/stats_info['Spon_Frames']
stats_info['OD_freq'] = stats_info['OD_repeats']*1.301/stats_info['Spon_Frames']
stats_info['Orien_freq'] = stats_info['Orien_repeats']*1.301/stats_info['Spon_Frames']
stats_info['Color_freq'] = stats_info['Color_repeats']*1.301/stats_info['Spon_Frames']
# save results.
ot.Save_Variable(r'D:\ZR\_Data_Temp\_Stats','Before_Super_ISI_all_V2',stats_info)
#%% get each recover map.
all_recover_similarity = pd.DataFrame(0,columns= ['Orien0_corr','Orien45_corr','Orien90_corr','Orien135_corr'],index = all_path)
for i,cp in enumerate(all_path):
    ac = ot.Load_Variable(cp,'Cell_Class.pkl')
    c_labels = stats_info.loc[cp,'Spon_SVM_Train'][0]
    spon_frame = ot.Load_Variable(cp,'Spon_Before.pkl')
    # get-0 graphs
    Orien0 = ac.Orien_t_graphs['Orien0-0'].loc['t_value']
    Orien45 = ac.Orien_t_graphs['Orien45-0'].loc['t_value']
    Orien90 = ac.Orien_t_graphs['Orien90-0'].loc['t_value']
    Orien135 = ac.Orien_t_graphs['Orien135-0'].loc['t_value']
    Orien0_frames = np.where(c_labels == 9)[0]
    if len(Orien0_frames)>0:
        Orien0_recover = spon_frame.iloc[Orien0_frames,:].mean(0)
        Orien0_corr,_ = pearsonr(Orien0,Orien0_recover)
    else:
        Orien0_corr = -1
    Orien45_frames = np.where(c_labels == 11)[0]
    if len(Orien45_frames)>0:
        Orien45_recover = spon_frame.iloc[Orien45_frames,:].mean(0)
        Orien45_corr,_ = pearsonr(Orien45,Orien45_recover)
    else:
        Orien45_corr = -1
    Orien90_frames = np.where(c_labels == 13)[0]
    if len(Orien90_frames)>0:
        Orien90_recover = spon_frame.iloc[Orien90_frames,:].mean(0)
        Orien90_corr,_ = pearsonr(Orien90,Orien90_recover)
    else:
        Orien90_corr = -1
    Orien135_frames = np.where(c_labels == 15)[0]
    if len(Orien135_frames)>0:
        Orien135_recover = spon_frame.iloc[Orien135_frames,:].mean(0)
        Orien135_corr,_ = pearsonr(Orien135,Orien135_recover)
    else:
        Orien135_corr = -1
    all_recover_similarity.iloc[i,:] = [Orien0_corr,Orien45_corr,Orien90_corr,Orien135_corr]
#%% Plot Repeat maps.
from scipy.signal import find_peaks
thres = 2
peak_height = 10
peak_dist = 5
determine_thres = 0.5
cp = all_path[1]
spon_series = ot.Load_Variable(cp,'Spon_Before.pkl')
c_on_series = np.array((spon_series>thres).sum(1))
peaks, height = find_peaks(c_on_series, height=peak_height, distance=peak_dist)
plt.switch_backend('webAgg')
plt.plot(c_on_series)
plt.plot(peaks, c_on_series[peaks], "x", color="red")
plt.show()