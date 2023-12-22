'''
This Graph discribe the repeat frequency of OD, Orien, Color 3 networks, using new class object, and regard consequtive series as a single repeat.

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
from Kill_Cache import kill_all_cache
from sklearn.model_selection import cross_val_score
from sklearn import svm
import umap
import umap.plot
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
from Cell_Class.UMAP_Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *

work_path = r'D:\_Path_For_Figs\_2312_ver2\Fig2'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Datas_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
#%%########################## STEP1, GET ALL NETWORK FREQUENCY #########################
spon_repeat_count = pd.DataFrame(columns=['Loc','Network','Repeat_Freq','Data'])
for i,c_loc in tqdm(enumerate(all_path_dic)):
    c_loc_last = c_loc.split('\\')[-1]
    # calculate real network repeat freq.
    c_spon_frame = ot.Load_Variable(c_loc,'Spon_Before.pkl')
    c_ac = ot.Load_Variable_v2(c_loc,'Cell_Class.pkl')
    c_reducer = ot.Load_Variable(c_loc,'All_Stim_UMAP_3D_20comp.pkl')
    c_analyzer = UMAP_Analyzer(ac = c_ac,umap_model=c_reducer,spon_frame=c_spon_frame)
    c_analyzer.Similarity_Compare_Average()
    c_spon_series = c_analyzer.spon_label
    tc_analyzer = Series_TC_info(input_series=c_spon_series)
    c_od_freq,c_orien_freq,c_color_freq = tc_analyzer.Freq_Estimation(type='Event')
    # and shuffle network repeat freq.
    shuffle_times = 10
    repeat_freq_s = np.zeros(shape = (shuffle_times,3),dtype = 'f8')# save in sequence od,orien,color.
    for j in range(shuffle_times):# shuffle
        spon_frame_s = Spon_Shuffler(c_spon_frame)
        spon_embedding_s = c_reducer.transform(spon_frame_s)
        c_spon_label_s = SVC_Fit(c_analyzer.svm_classifier,data = spon_embedding_s,thres_prob = 0)
        tc_analyzer_s = Series_TC_info(input_series=c_spon_label_s)
        c_od_freq_s,c_orien_freq_s,c_color_freq_s = tc_analyzer_s.Freq_Estimation(type='Event')
        repeat_freq_s[j,:] = [c_od_freq_s,c_orien_freq_s,c_color_freq_s]
    repeat_freq_s = repeat_freq_s.mean(0)
    spon_repeat_count.loc[len(spon_repeat_count),:] = [c_loc_last,'Eye',c_od_freq,'Real']
    spon_repeat_count.loc[len(spon_repeat_count),:] = [c_loc_last,'Orien',c_orien_freq,'Real']
    spon_repeat_count.loc[len(spon_repeat_count),:] = [c_loc_last,'Color',c_color_freq,'Real']
    spon_repeat_count.loc[len(spon_repeat_count),:] = [c_loc_last,'Eye',repeat_freq_s[0],'Shuffle']
    spon_repeat_count.loc[len(spon_repeat_count),:] = [c_loc_last,'Orien',repeat_freq_s[1],'Shuffle']
    spon_repeat_count.loc[len(spon_repeat_count),:] = [c_loc_last,'Color',repeat_freq_s[2],'Shuffle']
#%%####################### FIG 2E ,VISUALIZATION ###########################
plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (2.5,5),dpi = 180)
ax.axhline(y = 0,color='gray', linestyle='--')
sns.boxplot(data = spon_repeat_count,x = 'Data',y = 'Repeat_Freq',hue = 'Network',ax = ax,showfliers = False)
ax.set_title('Stim-like Ensemble Repeat Frequency',size = 10)
ax.set_xlabel('')
ax.set_ylabel('Repeat Frequency(Hz)')
ax.legend(title = 'Network',fontsize = 8)
ax.set_xticklabels(['Real Data','Random Select'],size = 7)
plt.show()