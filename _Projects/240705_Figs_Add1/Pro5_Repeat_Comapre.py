'''
This part will check the co-activation of each network, and try to compare it with random?
'''

#%% Load in for example loc.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import cv2
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")


from Cell_Class.Stim_Calculators import Stim_Cells
from Cell_Class.Format_Cell import Cell
import OS_Tools_Kit as ot
from Cell_Class.Advanced_Tools import *
from Classifier_Analyzer import *

expt_folder = r'D:\_All_Spon_Data_V1\L76_18M_220902'
# save_path = r'D:\_GoogleDrive_Files\#Figs\240627_Figs_FF1\Fig1'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
c_spon = np.array(ot.Load_Variable(expt_folder,'Spon_Before.pkl'))
all_path_dic = list(ot.Get_Subfolders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
all_path_dic_v2 = list(ot.Get_Subfolders(r'D:\_All_Spon_Data_V2'))
spon_dff = ac.Get_dFF_Frames('1-001',0.1,8500,13852)
#%%
N_Shuffle = 500
overlap_frame = pd.DataFrame(columns = ['Loc','OD_Orien','OD_Color','Orien_Color','Data_Type'])
for i,cloc in enumerate(all_path_dic):
    c_repeat = ot.Load_Variable(cloc,'All_Spon_Repeats_PCA10.pkl')
    cloc_name = cloc.split('\\')[-1]
    series_len = len(c_repeat)
    c_od = c_repeat['OD']>0
    c_orien = c_repeat['Orien']>0
    c_color = c_repeat['Color']>0
    od_orien_overlap = (c_od*c_orien).sum()/series_len
    od_color_overlap = (c_color*c_od).sum()/series_len
    orien_color_overlap = (c_color*c_orien).sum()/series_len
    overlap_frame.loc[len(overlap_frame)] = [cloc_name,od_orien_overlap,od_color_overlap,orien_color_overlap,'Real Data']
    for j in tqdm(range(N_Shuffle)):
        _,od_len = Label_Event_Cutter(c_od)
        shuffled_od = Random_Series_Generator(series_len,od_len)
        _,orien_len = Label_Event_Cutter(c_orien)
        shuffled_orien = Random_Series_Generator(series_len,orien_len)
        _,color_len = Label_Event_Cutter(c_color)
        shuffled_color = Random_Series_Generator(series_len,color_len)
        od_orien_overlap_s = (shuffled_od*shuffled_orien).sum()/series_len
        od_color_overlap_s = (shuffled_color*shuffled_od).sum()/series_len
        orien_color_overlap_s = (shuffled_color*shuffled_orien).sum()/series_len
        overlap_frame.loc[len(overlap_frame)] = [cloc_name,od_orien_overlap_s,od_color_overlap_s,orien_color_overlap_s,'Shuffled Data']
#%% Generate summary stats.
# all_loc_name = list(set(overlap_frame['Loc']))
all_prop_frame = pd.DataFrame(columns = ['Loc','Prop.','Network_Type'])
for i,cloc in enumerate(all_path_dic):
    c_repeat = ot.Load_Variable(cloc,'All_Spon_Repeats_PCA10.pkl')
    cloc_name = cloc.split('\\')[-1]
    series_len = len(c_repeat)
    c_od = c_repeat['OD']>0
    c_orien = c_repeat['Orien']>0
    c_color = c_repeat['Color']>0
    od_prop = c_od.sum()/series_len
    orien_prop = c_orien.sum()/series_len
    color_prop = c_color.sum()/series_len
    cloc_overlap_info = overlap_frame.loc[overlap_frame['Loc']==cloc_name]
    real = cloc_overlap_info.groupby('Data_Type').get_group('Real Data')
    od_orien = real['OD_Orien'].mean()
    od_color = real['OD_Color'].mean()
    orien_color = real['Orien_Color'].mean()
    shuffle = cloc_overlap_info.groupby('Data_Type').get_group('Shuffled Data')
    od_orien_s = shuffle['OD_Orien'].mean()
    od_color_s = shuffle['OD_Color'].mean()
    orien_color_s = shuffle['Orien_Color'].mean()
    # write into frames
    all_prop_frame.loc[len(all_prop_frame)] = [cloc_name,od_prop,'OD']
    all_prop_frame.loc[len(all_prop_frame)] = [cloc_name,orien_prop,'Orien']
    all_prop_frame.loc[len(all_prop_frame)] = [cloc_name,color_prop,'Color']
    all_prop_frame.loc[len(all_prop_frame)] = [cloc_name,od_orien,'OD-Orien']
    all_prop_frame.loc[len(all_prop_frame)] = [cloc_name,od_color,'OD-Color']
    all_prop_frame.loc[len(all_prop_frame)] = [cloc_name,orien_color,'Orien-Color']
    all_prop_frame.loc[len(all_prop_frame)] = [cloc_name,od_orien_s,'OD-Orien_S']
    all_prop_frame.loc[len(all_prop_frame)] = [cloc_name,od_color_s,'OD-Color_S']
    all_prop_frame.loc[len(all_prop_frame)] = [cloc_name,orien_color_s,'Orien-Color_S']

    

#%%      
plotable = all_prop_frame
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (12,5),dpi = 180)
# sns.histplot(data = plotable,hue = 'Loc',x = 'OD_Color',ax = ax,alpha = 0.5,common_norm=False,stat='percent',cbar=False)
# sns.stripplot(data=plotable, x="Network_Type", y="Prop.",hue = 'Loc')
sns.barplot(data=plotable, x="Network_Type", y="Prop.",capsize=0.2,width=0.5,palette=["C0", "C1", "C2", "C0", "C1", "C2","C0", "C1", "C2",])
#%% Do some stat test 
real = all_prop_frame.groupby('Network_Type').get_group('Orien_Color')
shuffle = all_prop_frame.groupby('Network_Type').get_group('Orien_Color_S')
t,p = stats.ttest_rel(real['Prop.'],shuffle['Prop.'])