'''
This script will generate traditional method spon event detection(Using thresed firing rate)
Compare this with UMAP result, will provide us the validity of conclusions.

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

all_path = ot.Get_Sub_Folders(r'D:\ZR\_Data_Temp\_All_Cell_Classes')
all_path = np.delete(all_path,[4,7]) # delete 2 point with not so good OD.
#%% get spon dics and target stim maps.(2OD,4Orien,6Color)
from scipy.signal import find_peaks
thres = 2
peak_height = 10
peak_dist = 5
determine_thres = 0.5

all_corr_frame = []
for i,cp in tqdm(enumerate(all_path)):
    c_spon = ot.Load_Variable(cp,'Spon_Before.pkl')
    c_on_series = np.array((c_spon>2).sum(1))
    peaks, height = find_peaks(c_on_series, height=peak_height, distance=peak_dist)
    # plt.switch_backend('webAgg')
    # plt.plot(c_on_series)
    # plt.plot(peaks, c_on_series[peaks], "x", color="red")
    # plt.show()
    ac = ot.Load_Variable(cp,'Cell_Class.pkl')
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
    c_Red = ac.Color_t_graphs['Red-0'].loc['t_value']
    c_Yellow = ac.Color_t_graphs['Yellow-0'].loc['t_value']
    c_Green = ac.Color_t_graphs['Green-0'].loc['t_value']
    c_Cyan = ac.Color_t_graphs['Cyan-0'].loc['t_value']
    c_Blue = ac.Color_t_graphs['Blue-0'].loc['t_value']
    c_Purple = ac.Color_t_graphs['Purple-0'].loc['t_value']
    # define a corr frame of all peaks.
    corr_frame = pd.DataFrame(0,index = ['LE','RE','Orien0','Orien22.5','Orien45','Orien67.5','Orien90','Orien112.5','Orien135','Orien157.5','Red','Yellow','Green','Cyan','Blue','Purple'],columns = peaks)
    for j,c_peak in enumerate(peaks):
        c_response = c_spon.iloc[c_peak,:]
        corr_frame.loc['LE',c_peak],_ = pearsonr(c_response,c_LE)
        corr_frame.loc['RE',c_peak],_ = pearsonr(c_response,c_RE)
        corr_frame.loc['Orien0',c_peak],_ = pearsonr(c_response,c_orien0)
        corr_frame.loc['Orien22.5',c_peak],_ = pearsonr(c_response,c_orien225)
        corr_frame.loc['Orien45',c_peak],_ = pearsonr(c_response,c_orien45)
        corr_frame.loc['Orien67.5',c_peak],_ = pearsonr(c_response,c_orien675)
        corr_frame.loc['Orien90',c_peak],_ = pearsonr(c_response,c_orien90)
        corr_frame.loc['Orien112.5',c_peak],_ = pearsonr(c_response,c_orien1125)
        corr_frame.loc['Orien135',c_peak],_ = pearsonr(c_response,c_orien135)
        corr_frame.loc['Orien157.5',c_peak],_ = pearsonr(c_response,c_orien1575)
        corr_frame.loc['Red',c_peak],_ = pearsonr(c_response,c_Red)
        corr_frame.loc['Yellow',c_peak],_ = pearsonr(c_response,c_Yellow)
        corr_frame.loc['Green',c_peak],_ = pearsonr(c_response,c_Green)
        corr_frame.loc['Cyan',c_peak],_ = pearsonr(c_response,c_Cyan)
        corr_frame.loc['Blue',c_peak],_ = pearsonr(c_response,c_Blue)
        corr_frame.loc['Purple',c_peak],_ = pearsonr(c_response,c_Purple)
        corr_frame.loc['All',c_peak],_ = pearsonr(c_response,c_All)
        
    all_corr_frame.append(corr_frame)

ot.Save_Variable(r'D:\ZR\_Data_Temp\_Stats','All_Corr_Peaks',all_corr_frame)
# %% get returned peak of all corrs.
corr_thres = 0.25 # if only corr>0.2,regard this as a repeat. Using the best repeatance.
near_thres = 0.0 # if top 2 corr are less than this, will be ignored.
# plt.switch_backend('webAgg')
# plt.hist(all_corr_frame[0].loc['RE',:],bins = 20)
# plt.show()
all_repeats = []
all_spon_length = []
for i,cp in tqdm(enumerate(all_path)):
    # meta info of all spon
    c_spon_frame = ot.Load_Variable(cp,'Spon_Before.pkl')
    c_on_series = np.array((c_spon_frame>2).sum(1))
    all_spon_length.append(len(c_on_series))
    # peak infos
    c_spon = all_corr_frame[i]
    max_corr = c_spon.max()
    max_value_indices = c_spon.idxmax()
    

    c_repeat = pd.DataFrame(0,columns = ['Peak','Repeat','Scale','Corr'],index = range(len(c_spon.columns)))
    for j,c_corr in enumerate(max_corr):
        c_peak_loc = max_corr.index[j]
        c_scale = c_on_series[c_peak_loc]
        second_max_corr = c_spon[c_peak_loc].nlargest(2).iloc[-1]
        if c_corr>corr_thres and (c_corr-second_max_corr)>near_thres: # a good return.
            c_repeat_stim = max_value_indices.iloc[j]
        else:
            c_repeat_stim = 'No_Stim'
            c_corr =0
        c_repeat.iloc[j,:] = [c_peak_loc,c_repeat_stim,c_scale,c_corr]
    all_repeats.append(c_repeat)
ot.Save_Variable(r'D:\ZR\_Data_Temp\_Stats','All_Traditional_Repeats',all_repeats)
#%% Count average repeatance of each kind of stims.
stats_info = pd.DataFrame(0,columns=['Spon_Frame_Num','Peak_Num','OD_peak','Orien_peak','Color_peak','Non_peak','All_peak'],index=all_path)
all_recover_similarity = pd.DataFrame(0,columns= ['LE_corr','RE_corr','Orien0_corr','Orien45_corr','Orien90_corr','Orien135_corr'],index = all_path)

for i,cp in tqdm(enumerate(all_path)):
    c_spon_frame = ot.Load_Variable(cp,'Spon_Before.pkl')
    c_framenum = all_spon_length[i]
    c_peaknum = all_repeats[i].shape[0]
    c_repeat = all_repeats[i]
    # count each 
    c_od_peaks = np.sum(((c_repeat['Repeat'] == 'LE')+(c_repeat['Repeat'] == 'RE'))>0)
    c_orien_peaks = np.sum(((c_repeat['Repeat'] == 'Orien0')+(c_repeat['Repeat'] == 'Orien22.5')+(c_repeat['Repeat'] == 'Orien45')+(c_repeat['Repeat'] == 'Orien67.5')+(c_repeat['Repeat'] == 'Orien90')+(c_repeat['Repeat'] == 'Orien112.5')+(c_repeat['Repeat'] == 'Orien135')+(c_repeat['Repeat'] == 'Orien157.5'))>0)
    c_color_peaks = np.sum(((c_repeat['Repeat'] == 'Red')+(c_repeat['Repeat'] == 'Yellow')+(c_repeat['Repeat'] == 'Green')+(c_repeat['Repeat'] == 'Cyan')+(c_repeat['Repeat'] == 'Blue')+(c_repeat['Repeat'] == 'Purple'))>0)
    c_non_peaks = np.sum(c_repeat['Repeat']=='No_Stim')
    c_all_peaks = np.sum(c_repeat['Repeat']=='All')
    stats_info.loc[cp] = [c_framenum,c_peaknum,c_od_peaks,c_orien_peaks,c_color_peaks,c_non_peaks,c_all_peaks]
    # get all stim graphs
    ac = ot.Load_Variable(cp,'Cell_Class.pkl')
    c_LE = ac.OD_t_graphs['L-0'].loc['t_value']
    c_RE = ac.OD_t_graphs['R-0'].loc['t_value']
    c_orien0 = ac.Orien_t_graphs['Orien0-0'].loc['t_value']
    c_orien45 = ac.Orien_t_graphs['Orien45-0'].loc['t_value']
    c_orien90 = ac.Orien_t_graphs['Orien90-0'].loc['t_value']
    c_orien135 = ac.Orien_t_graphs['Orien135-0'].loc['t_value']
    # c_Red = ac.Color_t_graphs['Red-0'].loc['t_value']
    # c_Yellow = ac.Color_t_graphs['Yellow-0'].loc['t_value']
    # c_Green = ac.Color_t_graphs['Green-0'].loc['t_value']
    # c_Cyan = ac.Color_t_graphs['Cyan-0'].loc['t_value']
    # c_Blue = ac.Color_t_graphs['Blue-0'].loc['t_value']
    # c_Purple = ac.Color_t_graphs['Purple-0'].loc['t_value']
    # get each recover graph.
    c_LE_peaks = c_repeat[c_repeat['Repeat'] == 'LE']
    if len(c_LE_peaks) != 0:
        c_LE_recover = c_spon_frame.iloc[c_LE_peaks['Peak'],:].mean(0)
        c_LE_corr,_ = stats.pearsonr(c_LE,c_LE_recover)
    else:
        c_LE_corr = -1
    c_RE_peaks = c_repeat[c_repeat['Repeat'] == 'RE']
    if len(c_RE_peaks) != 0:
        c_RE_recover = c_spon_frame.iloc[c_RE_peaks['Peak'],:].mean(0)
        c_RE_corr,_ = stats.pearsonr(c_RE,c_RE_recover)
    else:
        c_RE_corr = -1
    c_Orien0_peaks = c_repeat[c_repeat['Repeat'] == 'Orien0']
    if len(c_Orien0_peaks) != 0:
        c_Orien0_recover = c_spon_frame.iloc[c_Orien0_peaks['Peak'],:].mean(0)
        c_Orien0_corr,_ = stats.pearsonr(c_orien0,c_Orien0_recover)
    else:
        c_Orien0_corr = -1
    c_Orien45_peaks = c_repeat[c_repeat['Repeat'] == 'Orien45']
    if len(c_Orien45_peaks) != 0:
        c_Orien45_recover = c_spon_frame.iloc[c_Orien45_peaks['Peak'],:].mean(0)
        c_Orien45_corr,_ = stats.pearsonr(c_orien45,c_Orien45_recover)
    else:
        c_Orien45_corr = -1
    c_Orien90_peaks = c_repeat[c_repeat['Repeat'] == 'Orien90']
    if len(c_Orien90_peaks) != 0:
        c_Orien90_recover = c_spon_frame.iloc[c_Orien90_peaks['Peak'],:].mean(0)
        c_Orien90_corr,_ = stats.pearsonr(c_orien90,c_Orien90_recover)
    else:
        c_Orien90_corr = -1
    c_Orien135_peaks = c_repeat[c_repeat['Repeat'] == 'Orien135']
    if len(c_Orien135_peaks) != 0:
        c_Orien135_recover = c_spon_frame.iloc[c_Orien135_peaks['Peak'],:].mean(0)
        c_Orien135_corr,_ = stats.pearsonr(c_orien135,c_Orien135_recover)
    else:
        c_Orien135_corr = -1
    
    all_recover_similarity.loc[cp,:] = [c_LE_corr,c_RE_corr,c_Orien0_corr,c_Orien45_corr,c_Orien90_corr,c_Orien135_corr]
# get propotion of OD frame, Orien frame, color frame an all frame.
stats_info['All_Peak_Frequency'] = stats_info['Peak_Num']*1.301/stats_info['Spon_Frame_Num']
stats_info['OD_Peak_Frequency'] = stats_info['OD_peak']*1.301/stats_info['Spon_Frame_Num']
stats_info['Orien_Peak_Frequency'] = stats_info['Orien_peak']*1.301/stats_info['Spon_Frame_Num']
stats_info['Color_Peak_Frequency'] = stats_info['Color_peak']*1.301/stats_info['Spon_Frame_Num']
stats_info['Non_Peak_Frequency'] = stats_info['Non_peak']*1.301/stats_info['Spon_Frame_Num']
#%% Plotting propotion
melted_df = pd.melt(stats_info, value_vars=['All_Peak_Frequency','OD_Peak_Frequency','Orien_Peak_Frequency','Color_Peak_Frequency','Non_Peak_Frequency'], var_name='Type',value_name ='Repeat Frequency (Hz)')
plt.switch_backend('webAgg')
sns.barplot(data = melted_df,y = 'Repeat Frequency (Hz)',x = 'Type')
plt.show()

#%% Plotting recover corr
melted_frame = pd.melt(all_recover_similarity,value_vars=['LE_corr','RE_corr','Orien0_corr','Orien45_corr','Orien90_corr','Orien135_corr'],value_name='recover map r',var_name = 'Map Name')
selected_frame = melted_frame[melted_frame['recover map r']!= -1]
plt.switch_backend('webAgg')
sns.barplot(data = selected_frame,y = 'recover map r',x = 'Map Name',capsize=.2)
plt.show()
#%% What if we count frame instead of event?
