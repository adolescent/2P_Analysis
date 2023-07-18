'''
This script will discuss the independency of different networks. Including Orien vs OD,LE vs RE, Orien0 vs 90 etc


########## Series Find Method ###########
1.Force discrimination by UMAP
2.Correlation with stim maps.
3.Weighted tuning curve on cell response.

############# Determine method ###################
1.GCA
2.Slide window corr
3.Random scale with 

################## Preprocess ################
Subtract cortex activation.
Normalize series by activation.
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

all_path = ot.Get_Sub_Folders(r'D:\ZR\_Data_Temp\_All_V1_Before_Cell_Classes')
all_path = np.delete(all_path,[4,7]) # delete 2 point with not so good OD.
stats_path = r'D:\ZR\_Data_Temp\_Stats'
cp = all_path[0]
#%% slide window cortex activation.
on_thres = 0
win_size = 300
win_step = 100
ac = ot.Load_Variable(cp,'Cell_Class.pkl')
spon_frame = ot.Load_Variable(cp,'Spon_Before.pkl')
# on_series = np.array((spon_frame>on_thres).sum(1))
def Plotter(series):
    plt.switch_backend('webAgg')
    plt.plot(series)
    plt.show()
def Slide_Win_Avr(input_series,win_size,win_step,method = 'sum'):
    win_num = (len(input_series)-win_size)//win_step+1
    avr_series = np.zeros(win_num)
    for i in range(win_num):
        c_win = input_series[i*win_step:win_size+i*win_step]
        if method == 'sum':
            avr_series[i] = c_win.sum()
        elif method == 'avr':
            avr_series[i] = c_win.mean()
    return avr_series
#%% Direct corr method.
all_corrs = pd.DataFrame(0,columns=['LR','LR_Template','HV','HV_Template','AO','AO_Template'],index = all_path)
for i,cp in tqdm(enumerate(all_path)):
# get all stim graphs.
    ac = ot.Load_Variable(cp,'Cell_Class.pkl')
    spon_frame = ot.Load_Variable(cp,'Spon_Before.pkl')
    c_LE = ac.OD_t_graphs['L-0'].loc['t_value']
    c_RE = ac.OD_t_graphs['R-0'].loc['t_value']
    c_All = ac.Orien_t_graphs['All-0'].loc['t_value']
    c_orien0 = ac.Orien_t_graphs['Orien0-0'].loc['t_value']
    c_orien225 = ac.Orien_t_graphs['Orien22.5-0'].loc['t_value']
    c_orien45 = ac.Orien_t_graphs['Orien45-0'].loc['t_value']
    c_orien675 = ac.Orien_t_graphs['Orien67.5-0'].loc['t_value']
    c_orien90 = ac.Orien_t_graphs['Orien90-0'].loc['t_value']
    c_orien1125 = ac.Orien_t_graphs['Orien112.5-0'].loc['t_value']
    c_orien135 = ac.Orien_t_graphs['Orien135-0'].loc['t_value']
    c_orien1575 = ac.Orien_t_graphs['Orien157.5-0'].loc['t_value']
# and correlation train between Stim graphs and spon series.
    frame_num = spon_frame.shape[0]
    LE_corr = np.zeros(frame_num)
    RE_corr = np.zeros(frame_num)
    Orien0_corr = np.zeros(frame_num)
    Orien45_corr = np.zeros(frame_num)
    Orien90_corr = np.zeros(frame_num)
    Orien135_corr = np.zeros(frame_num)
    All_corr = np.zeros(frame_num)
    for i in tqdm(range(frame_num)):
        c_spon = np.array(spon_frame.iloc[i,:])
        LE_corr[i],_ = stats.pearsonr(c_spon,c_LE)
        RE_corr[i],_ = stats.pearsonr(c_spon,c_RE)
        Orien0_corr[i],_ = stats.pearsonr(c_spon,c_orien0)
        Orien45_corr[i],_ = stats.pearsonr(c_spon,c_orien45)
        Orien90_corr[i],_ = stats.pearsonr(c_spon,c_orien90)
        Orien135_corr[i],_ = stats.pearsonr(c_spon,c_orien135)
        All_corr[i],_ = stats.pearsonr(c_spon,c_All)
    c_LR,_ = stats.pearsonr(LE_corr,RE_corr)
    c_LR_template,_ = stats.pearsonr(c_LE,c_RE)
    c_HV,_ = stats.pearsonr(Orien0_corr,Orien90_corr)
    c_HV_template,_ = stats.pearsonr(c_orien0,c_orien90)
    c_AO,_ = stats.pearsonr(Orien45_corr,Orien135_corr)
    c_AO_template,_ = stats.pearsonr(c_orien45,c_orien135)
    all_corrs.loc[cp,:] = [c_LR,c_LR_template,c_HV,c_HV_template,c_AO,c_AO_template]
#%% Plot stast results.
melted_frame = pd.melt(all_corrs,value_vars=['LR','LR_Template','HV','HV_Template','AO','AO_Template'],value_name='Pearson r',var_name = 'Map Name')
selected_frame = melted_frame[melted_frame['Pearson r']!= -1]
plt.switch_backend('webAgg')
ax = sns.barplot(data = selected_frame,y = 'Pearson r',x = 'Map Name',capsize=.2)
sns.stripplot(data=selected_frame, y = 'Pearson r',x = 'Map Name',ax = ax,color = 'black')
plt.show()
    