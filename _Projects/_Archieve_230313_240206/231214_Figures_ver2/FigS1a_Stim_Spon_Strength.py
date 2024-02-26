'''
This graph discribe the strength difference between stim and spon.
SD,Mean,Max are contained.

Be cautious of dff calculation method.

'''
#%% Import and initialization
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
from Filters import Signal_Filter


work_path = r'D:\_Path_For_Figs\_2312_ver2\Fig1'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Datas_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)

#%%##################### FUNCTIONS ############################
def dFF(F_series,method = 'least',prop=0.1): # dFF method can be changed here.
    if method == 'least':
        base_num = int(len(F_series)*prop)
        base_id = np.argpartition(F_series, base_num)[:base_num]
        base = F_series[base_id].mean()
    dff_series = (F_series-base)/base
    return dff_series,base

#%%################# STEP1, GENERATE DF/F SERIES###############################
Cell_Response_Frame = pd.DataFrame(columns=['Loc','Cell','Type','Part','Mean','Std','Max','Base','Disp'])
filter_para = (0.05*2/1.301,0.3*2/1.301)

for i,c_loc in tqdm(enumerate(all_path_dic)):
    ac = ot.Load_Variable_v2(c_loc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable(c_loc,'Spon_Before.pkl')
    spon_start = c_spon.index[0]
    acd = ac.all_cell_dic
    sfa = ac.Stim_Frame_Align
    all_cell_name = list(acd.keys())
    od_stim = np.array(sfa['Run'+ac.odrun[2:]]['Original_Stim_Train'])
    orien_stim = np.array(sfa['Run'+ac.orienrun[2:]]['Original_Stim_Train'])
    color_stim = np.array(sfa['Run'+ac.colorrun[2:]]['Original_Stim_Train'])
    # for current location, cycle all cells.
    for j,cc in enumerate(all_cell_name):
        cc_dic = acd[cc]
        # filt all series first.
        c_spon_series = Signal_Filter(cc_dic['1-001'][spon_start:],order=7,filter_para=filter_para)
        c_od_series = Signal_Filter(cc_dic[ac.odrun],order=7,filter_para=filter_para)
        c_orien_series = Signal_Filter(cc_dic[ac.orienrun],order=7,filter_para=filter_para)
        c_color_series = Signal_Filter(cc_dic[ac.colorrun],order=7,filter_para=filter_para)
        # then get df/f by function above.
        c_spon_dff,base_spon = dFF(c_spon_series)
        c_od_dff,base_od = dFF(c_od_series)
        c_orien_dff,base_orien = dFF(c_orien_series)
        c_color_dff,base_color = dFF(c_color_series)
        # and seperate on and off.
        c_od_on = c_od_dff[np.where(od_stim != -1)[0]]
        c_od_off = c_od_dff[np.where(od_stim == -1)[0]]
        c_orien_on = c_orien_dff[np.where(orien_stim != -1)[0]]
        c_orien_off = c_orien_dff[np.where(orien_stim == -1)[0]]
        c_color_on = c_color_dff[np.where(color_stim != -1)[0]]
        c_color_off = c_color_dff[np.where(color_stim == -1)[0]]
        # write them into a pandas frame.
        ## ODs
        Cell_Response_Frame.loc[len(Cell_Response_Frame),:] = [c_loc.split('\\')[-1],cc,'Spon','All',c_spon_dff.mean(),c_spon_dff.std(),c_spon_dff.max(),base_spon,c_spon_dff]
        Cell_Response_Frame.loc[len(Cell_Response_Frame),:] = [c_loc.split('\\')[-1],cc,'OD','All',c_od_dff.mean(),c_od_dff.std(),c_od_dff.max(),base_od,c_od_dff]
        Cell_Response_Frame.loc[len(Cell_Response_Frame),:] = [c_loc.split('\\')[-1],cc,'OD','ON',c_od_on.mean(),c_od_on.std(),c_od_on.max(),base_od,c_od_on]
        Cell_Response_Frame.loc[len(Cell_Response_Frame),:] = [c_loc.split('\\')[-1],cc,'OD','OFF',c_od_off.mean(),c_od_off.std(),c_od_off.max(),base_od,c_od_off]
        # use model to save time.
        # Orientations
        target = c_orien_dff
        Cell_Response_Frame.loc[len(Cell_Response_Frame),:] = [c_loc.split('\\')[-1],cc,'Orien','All',target.mean(),target.std(),target.max(),base_orien,target]
        target = c_orien_on
        Cell_Response_Frame.loc[len(Cell_Response_Frame),:] = [c_loc.split('\\')[-1],cc,'Orien','ON',target.mean(),target.std(),target.max(),base_orien,target]
        target = c_orien_off
        Cell_Response_Frame.loc[len(Cell_Response_Frame),:] = [c_loc.split('\\')[-1],cc,'Orien','OFF',target.mean(),target.std(),target.max(),base_orien,target]
        # Colors
        target = c_color_dff
        Cell_Response_Frame.loc[len(Cell_Response_Frame),:] = [c_loc.split('\\')[-1],cc,'Color','All',target.mean(),target.std(),target.max(),base_color,target]
        target = c_color_on
        Cell_Response_Frame.loc[len(Cell_Response_Frame),:] = [c_loc.split('\\')[-1],cc,'Color','ON',target.mean(),target.std(),target.max(),base_color,target]
        target = c_color_off
        Cell_Response_Frame.loc[len(Cell_Response_Frame),:] = [c_loc.split('\\')[-1],cc,'Color','OFF',target.mean(),target.std(),target.max(),base_color,target]
ot.Save_Variable(work_path,'FigS1a_dFF_comare',Cell_Response_Frame)
#%%###################### STEP2, GENERATE COMPARE MATRIX. ##############################
# get spon and stim groups.
all_spon = Cell_Response_Frame.groupby('Type').get_group('Spon')
orien_all = Cell_Response_Frame.groupby('Type').get_group('Orien').groupby('Part').get_group('All')
orien_on = Cell_Response_Frame.groupby('Type').get_group('Orien').groupby('Part').get_group('ON')
orien_off = Cell_Response_Frame.groupby('Type').get_group('Orien').groupby('Part').get_group('OFF')
# save compare situation in a pandas frame.
Compare_Frame = pd.DataFrame(index = range(len(all_spon)*3),columns=['Spon','Stim All','Stim ON','Stim ISI','Method'])
for i in range(len(all_spon)):
    Compare_Frame.loc[i,:] = [np.array(all_spon['Mean'])[i],np.array(orien_all['Mean'])[i],np.array(orien_on['Mean'])[i],np.array(orien_off['Mean'])[i],'Mean']
    Compare_Frame.loc[i+len(all_spon),:] = [np.array(all_spon['Std'])[i],np.array(orien_all['Std'])[i],np.array(orien_on['Std'])[i],np.array(orien_off['Std'])[i],'Std']
    Compare_Frame.loc[i+2*len(all_spon),:] = [np.array(all_spon['Max'])[i],np.array(orien_all['Max'])[i],np.array(orien_on['Max'])[i],np.array(orien_off['Max'])[i],'Max']

#%%######################## STEP3, PLOT COMPARE GRAPH ###############################

avr_data = Compare_Frame.groupby('Method').get_group('Mean')
std_data = Compare_Frame.groupby('Method').get_group('Std')
max_data = Compare_Frame.groupby('Method').get_group('Max')
avr_range = [0,1.5]
std_range = [0,1]
max_range = [0,5]

# we plot only mean response, but std and max response can be plot the same way.
plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12,9), gridspec_kw={'height_ratios': [1,1.5]},sharex = False)
# first, scatter plots
sns.scatterplot(data = avr_data,x = 'Spon',y = 'Stim All',ax = axes[0,0],s = 2)
sns.lineplot(x = avr_range,y = avr_range,ax = axes[0,0],linestyle='--',c = 'gray')
axes[0,0].set_title('Mean Response Spon vs Whole Stim Series',fontsize = 10)

sns.scatterplot(data = avr_data,x = 'Spon',y = 'Stim ON',ax = axes[0,1],s = 2)
sns.lineplot(x = avr_range,y = avr_range,ax = axes[0,1],linestyle='--',c = 'gray')
axes[0,1].set_title('Mean Response Spon vs Stim Present',fontsize = 10)

sns.scatterplot(data = avr_data,x = 'Spon',y = 'Stim ISI',ax = axes[0,2],s = 2)
sns.lineplot(x = avr_range,y = avr_range,ax = axes[0,2],linestyle='--',c = 'gray')
axes[0,2].set_title('Mean Response Stim vs Stim ISI',fontsize = 10,)

# second, hist plot.
melted_avr_all = pd.melt(avr_data,value_vars=['Spon','Stim All'],value_name='Mean Response',var_name='Run type')
melted_avr_on = pd.melt(avr_data,value_vars=['Spon','Stim ON'],value_name='Mean Response',var_name='Run type')
melted_avr_isi = pd.melt(avr_data,value_vars=['Spon','Stim ISI'],value_name='Mean Response',var_name='Run type')
sns.histplot(data = melted_avr_all,x ='Mean Response',hue = 'Run type',ax = axes[1,0],alpha = 0.7,common_norm= False,stat="density")
sns.histplot(data = melted_avr_on,x ='Mean Response',hue = 'Run type',ax = axes[1,1],alpha = 0.7,common_norm= False,stat="density")
sns.histplot(data = melted_avr_isi,x ='Mean Response',hue = 'Run type',ax = axes[1,2],alpha = 0.7,common_norm= False,stat="density")

for i in range(3):
    axes[0,i].set_xlim(avr_range)
    axes[0,i].set_ylim(avr_range)
    axes[1,i].set_xlim(avr_range)

axes[1,1].sharey(axes[1,0])
axes[1,2].sharey(axes[1,0])

plt.suptitle('Cell Response In Spontaneous and In Stimulus',fontsize = 20)
plt.tight_layout()
plt.show()