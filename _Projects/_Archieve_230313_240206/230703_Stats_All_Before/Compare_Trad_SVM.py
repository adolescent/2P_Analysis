'''
This script will compare traditional result with SVM seperation results. Trying to find the property of non tuning peaks in peak find, and get the columns' impacts.

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
stats_path = r'D:\ZR\_Data_Temp\_Stats'
#%% get svm orientation peak but not in direct corr find.
from scipy.signal import find_peaks
thres = 2
peak_height = 10
peak_dist = 5
corr_thres = 0.4
all_peak_corrs = ot.Load_Variable(stats_path,'All_Corr_Peaks.pkl')
all_frame_corrs = ot.Load_Variable(stats_path,'All_Corr_Frames.pkl')
all_repeat_frame = []
# for i,cp in tqdm(enumerate(all_path)):
cp = all_path[3]
c_peak_corr = all_peak_corrs[3]
c_frame_corr = all_frame_corrs[3]
c_spon_frame = ot.Load_Variable(cp,'Spon_Before.pkl')
max_corr = c_peak_corr.max()
max_value_indices = c_peak_corr.idxmax()
# frame version here
# max_corr = c_frame_corr.max()
# max_value_indices = c_frame_corr.idxmax()
c_repeat_frame = pd.DataFrame(0,columns = ['Peak_Loc','Repeat_Stim'],index = range(c_peak_corr.shape[1]))
for j,c_corr in enumerate(max_corr):
    c_peak_loc = max_corr.index[j]
    if c_corr>corr_thres: # a good repeat
        c_repeat = max_value_indices.iloc[j]
    else:
        c_repeat = 'No_Stim'
    c_repeat_frame.loc[j] = [c_peak_loc,c_repeat]
all_repeat_frame.append(c_repeat_frame)
orien_peaks = c_repeat_frame[(c_repeat_frame['Repeat_Stim']=='Orien0')|(c_repeat_frame['Repeat_Stim']=='Orien22.5')|(c_repeat_frame['Repeat_Stim']=='Orien45')|(c_repeat_frame['Repeat_Stim']=='Orien67.5')|(c_repeat_frame['Repeat_Stim']=='Orien90')|(c_repeat_frame['Repeat_Stim']=='Orien112.5')|(c_repeat_frame['Repeat_Stim']=='Orien135')|(c_repeat_frame['Repeat_Stim']=='Orien157.5')]
od_peaks = c_repeat_frame[(c_repeat_frame['Repeat_Stim']=='LE')|(c_repeat_frame['Repeat_Stim']=='RE')]

c_on_series = np.array((c_spon_frame>2).sum(1))
peaks, height = find_peaks(c_on_series, height=peak_height, distance=peak_dist)
c_svm_labels = ot.Load_Variable(cp,'SVM_Unsup_Labels.pkl')
svm_on_series = ((c_svm_labels>0)*(c_svm_labels<17))
svm_orien_series = ((c_svm_labels>8)*(c_svm_labels<17))
svm_od_series = ((c_svm_labels>0)*(c_svm_labels<9))
od_events,od_lens = Label_Event_Cutter(svm_od_series)
orien_events,orien_lens = Label_Event_Cutter(svm_orien_series)
on_events,on_lens = Label_Event_Cutter(svm_on_series)

# plotter, used for quality check.
plt.switch_backend('webAgg')
plt.plot(c_on_series)
# indices = [i for i, x in enumerate(svm_on_series) if x]
# plt.barh(y=0, width=indices, height=-1, color='red')
plt.plot(orien_peaks['Peak_Loc'],c_on_series[orien_peaks['Peak_Loc']],'s',color = 'blue')
plt.plot(od_peaks['Peak_Loc'],c_on_series[od_peaks['Peak_Loc']],'s',color = 'Yellow')
# for j,label in enumerate(svm_orien_series):
#     if label == True:
#         plt.plot(j,c_on_series[j],'o',color = 'red')
for j,c_events in enumerate(orien_events):
    plt.plot(c_events[0],c_on_series[c_events[0]],'o',color = 'red')
plt.plot(peaks, c_on_series[peaks], "x", color="black")
plt.show()
#%% 
# ac = ot.Load_Variable(cp,'Cell_Class.pkl')
c_frame = c_spon_frame.iloc[7060,:]
plt.switch_backend('webAgg')
sns.heatmap(ac.Generate_Weighted_Cell(weight = c_frame),center = 0,square = True, xticklabels=False, yticklabels=False)
plt.show()
#%% corr list between 6025-6030 vs orientation map 112.5
c_graph_1125 = ac.Orien_t_graphs['Orien135-0'].loc['t_value']
frame_lists = c_spon_frame.iloc[7055:7061,:]
corrs = np.zeros(6)
for i in range(6):
    c_corr,_ = stats.pearsonr(frame_lists.iloc[i,:],c_graph_1125)
    corrs[i] = c_corr
plt.switch_backend('webAgg')
plt.plot(range(7055,7061),corrs)
plt.show()
#%% Let's combine spon event if it's head/tail is 5 frame near nearest event.
link_dist = 5
peak_locs = np.array(c_peak_corr.columns)
real_oriens = []
for i in range(len(orien_events)-1):
    if (orien_events[i+1][-1] - orien_events[i][0])>link_dist:
        real_oriens.append(orien_events[i])
real_ods = []
for i in range(len(od_events)-1):
    if (od_events[i+1][-1] - od_events[i][0])>link_dist:
        real_ods.append(od_events[i])

#%% Using Raw data svm.
raw_svm_series = []
for i,cp in tqdm(enumerate(all_path)):
    c_spon = ot.Load_Variable(cp,'Spon_Before.pkl')
    ac = ot.Load_Variable(cp,'Cell_Class.pkl')
    all_stim_frame,all_stim_label = ac.Combine_Frame_Labels(od = True,orien = True,color = True,isi = True)
    c_umap_svm_labels = ot.Load_Variable(cp,'SVM_Unsup_Labels.pkl')
    classifier,score = SVM_Classifier(embeddings=all_stim_frame,label = all_stim_label)
    c_raw_svm_series = SVC_Fit(classifier,c_spon,thres_prob = 0)

    raw_svm_on_series = ((c_raw_svm_series>0)*(c_raw_svm_series<17))
    raw_svm_orien_series = ((c_raw_svm_series>8)*(c_raw_svm_series<17))
    raw_svm_od_series = ((c_raw_svm_series>0)*(c_raw_svm_series<9))
    raw_svm_color_series = ((c_raw_svm_series>16))
    raw_svm_series.append(c_raw_svm_series)
ot.Save_Variable(stats_path,'Raw_SVM_series',raw_svm_series)
#%% calculate repeats
raw_stats_info = pd.DataFrame(0,columns = ['Spon_Num','On_Frame','OD_Frame','Orien_Frame','Color_Frame'],index = all_path)
for i,cp in enumerate(all_path):
    c_raw_series = raw_svm_series[i]
    c_spon_num = c_raw_series.shape[0]
    c_all_frame = np.sum(c_raw_series>0)
    c_od_frame = np.sum((c_raw_series>0)*(c_raw_series<9))
    c_orien_frame = np.sum((c_raw_series>8)*(c_raw_series<17))
    c_color_frame = np.sum((c_raw_series>0)*(c_raw_series>16))
    raw_stats_info.loc[cp,:] = [c_spon_num,c_all_frame,c_od_frame,c_orien_frame,c_color_frame]
raw_stats_info['All_Event'] = raw_stats_info['On_Frame']*1.301/raw_stats_info['Spon_Num']
raw_stats_info['OD_Event'] = raw_stats_info['OD_Frame']*1.301/raw_stats_info['Spon_Num']
raw_stats_info['Orien_Event'] = raw_stats_info['Orien_Frame']*1.301/raw_stats_info['Spon_Num']
raw_stats_info['Color_Event'] = raw_stats_info['Color_Frame']*1.301/raw_stats_info['Spon_Num']
#%% plot raw data svm results.
melted_df = pd.melt(raw_stats_info, value_vars=['All_Event','OD_Event','Orien_Event','Color_Event'], var_name='Type',value_name ='Repeat Frequency (Hz)')
plt.switch_backend('webAgg')
sns.barplot(data = melted_df,y = 'Repeat Frequency (Hz)',x = 'Type')
plt.show()

