'''
Define propriate threshold of SVM discrimitator.

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
from scipy.stats import pearsonr

wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220420_L91\_CAIMAN'
# wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220914_L85_2P\_CAIMAN'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
# reducer = ot.Load_Variable(wp,r'Stim_No_ISI_UMAP_Unsup_3d.pkl')
spon_frame = ac.Z_Frames['1-001']
od_frame = ac.Z_Frames[ac.odrun]
orien_frame = ac.Z_Frames[ac.orienrun]
all_frame,all_label = ac.Combine_Frame_Labels(isi = True)
#%% if load in 
reducer = ot.Load_Variable(wp,'Stim_No_ISI_UMAP_Unsup_3d.pkl')
u = reducer.embedding_
spon_embeddings = reducer.transform(spon_frame)
#%% reducer for isi included data.
kill_all_cache(r'C:\ProgramData\anaconda3\envs\umapzr')
reducer_full = umap.UMAP(n_components=3,n_neighbors=20)
reducer_full.fit(all_frame)
u = reducer_full.embedding_
spon_embeddings = reducer_full.transform(spon_frame)
ot.Save_Variable(wp,'Stim_All_UMAP_Unsup_3d',reducer_full)
#%% Get each -0 t graphs. 
LE = ac.OD_t_graphs['L-0'].loc['t_value']
RE = ac.OD_t_graphs['R-0'].loc['t_value']
Orien0 = ac.Orien_t_graphs['Orien0-0'].loc['t_value']
Orien45 = ac.Orien_t_graphs['Orien45-0'].loc['t_value']
Orien90 = ac.Orien_t_graphs['Orien90-0'].loc['t_value']
Orien135 = ac.Orien_t_graphs['Orien135-0'].loc['t_value']

#%% train svm with different parameters to determine best 
# svm_thres = [0,0.1,0.2,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9]
svm_thres = [0,0.1,0.2,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7]
svm_perform_frame = pd.DataFrame(0,columns= ['SVM_Thres','Repeat_Count','LE_corr','RE_corr','Orien0_corr','Orien45_corr','Orien90_corr','Orien135_corr'],index= range(len(svm_thres)))
classifier,score = SVM_Classifier(u,all_label,C = 10)
for i,c_thres in enumerate(svm_thres):
    # get correlation and test_score.
    c_labels = SVC_Fit(classifier=classifier,data = spon_embeddings,thres_prob= c_thres)
    c_labels[c_labels == 0] = -1
    repeat_count = np.sum(c_labels>0)
    LE_frames = np.where((c_labels>0)*(c_labels<9)*(c_labels%2 == 1))[0]
    if len(LE_frames)>0:
        LE_recover = spon_frame.iloc[LE_frames,:].mean(0)
        LE_corr,_ = pearsonr(LE,LE_recover)
    else:
        LE_corr = -1
    RE_frames = np.where((c_labels>0)*(c_labels<9)*(c_labels%2 == 0))[0]
    if len(RE_frames)>0:
        RE_recover = spon_frame.iloc[RE_frames,:].mean(0)
        RE_corr,_ = pearsonr(RE,RE_recover)
    else:
        RE_corr = -1
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
    svm_perform_frame.iloc[i,:] = [c_thres,repeat_count,LE_corr,RE_corr,Orien0_corr,Orien45_corr,Orien90_corr,Orien135_corr]
#%% Melt this data frame to plot error bar graph.
melted_frame = pd.melt(svm_perform_frame,id_vars=['SVM_Thres'],value_vars=['LE_corr','RE_corr','Orien0_corr','Orien45_corr','Orien90_corr','Orien135_corr'],value_name='repeat_corr')
selected_melted_frame = melted_frame[melted_frame['repeat_corr']!= -1]
#%% plot result above.
plt.switch_backend('webAgg')
fig,ax = plt.subplots(figsize = (12,10))
# ax = sns.lineplot(data = svm_perform_frame,x = 'SVM_Thres',y = 'Repeat_Count')
# ax = sns.lineplot(data = svm_perform_frame,x = 'SVM_Thres',y = 'LE_corr')
# ax = sns.lineplot(data = svm_perform_frame,x = 'SVM_Thres',y = 'RE_corr')
ax = sns.lineplot(data = selected_melted_frame,x = 'SVM_Thres',y = 'repeat_corr',err_style="bars", errorbar=("se", 2))
# ax = sns.lineplot(data = selected_melted_frame,x = 'SVM_Thres',y = 'repeat_corr',hue = 'variable')
plt.show()