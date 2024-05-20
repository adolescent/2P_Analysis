'''

This result will produce timecourse information of the given Time series, here we will discuss the waittime of graphs, overlapping toward randoms, so we will see the relationship with 3 different networks.


'''
#%% Initialization
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
from Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *


wp = r'D:\_Path_For_Figs\240312_Figs_v4_A1\A1_Waittime_Distribution'
expt_loc = r'D:\_All_Spon_Data_V1\L76_18M_220902'

all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
#%% Functions will use
def Start_Time_Finder(series):
        
    # Find the indices where the series switches from 0 to 1
    switch_indices = np.where(np.diff(series) == 1)[0] + 1 


    return switch_indices


#%% ####################### STEP0 - SAVE ALL REPEAT CURVES ###########################
pcnum = 10
all_spon_repeats = {}
for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    c_ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = np.array(ot.Load_Variable(cloc,'Spon_Before.pkl'))
    # Use spontaneous embedded pca model.
    spon_pcs,spon_coords,spon_models = Z_PCA(Z_frame=c_spon,sample='Frame',pcnum=pcnum)
    # orientation trains
    analyzer_orien = Classify_Analyzer(ac = c_ac,umap_model=spon_models,spon_frame=c_spon,od = 0, color = 0,orien = 1)
    analyzer_orien.Train_SVM_Classifier()
    spon_label_orien = analyzer_orien.spon_label
    # od trains
    analyzer_od = Classify_Analyzer(ac = c_ac,umap_model=spon_models,spon_frame=c_spon,od = 1, color = 0,orien = 0)
    analyzer_od.Train_SVM_Classifier()
    spon_label_od = analyzer_od.spon_label
    # color trains
    analyzer_color = Classify_Analyzer(ac = c_ac,umap_model=spon_models,spon_frame=c_spon,od = 0, color = 1,orien = 0)
    analyzer_color.Train_SVM_Classifier()
    spon_label_color = analyzer_color.spon_label
    # save all 3 trains into a pd frame.
    train_series = pd.DataFrame([spon_label_od,spon_label_orien,spon_label_color],index = ['OD','Orien','Color'],columns = range(len(spon_label_orien)))
    all_spon_repeats[cloc_name] = train_series

ot.Save_Variable(wp,'All_Repeat_Series',all_spon_repeats)



#%%##### 1. Wait time of Each Network & elapse time of each networks.#####
# example_frame = (all_spon_repeats['L76_18M_220902']>0).astype('f8')
# od_start_time = Start_Time_Finder(np.array(example_frame.loc['OD',:]))
# orien_start_time = Start_Time_Finder(np.array(example_frame.loc['Orien',:]))
# color_start_time = Start_Time_Finder(np.array(example_frame.loc['Color',:]))

# get nearest next network waittime.
all_network_waittime = pd.DataFrame(columns = ['Loc','Net_Before','Net_After','Waittime','Start_Time'])
for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    c_repeat_frame = all_spon_repeats[cloc_name]
    binned_repeat_frame = (c_repeat_frame>0).astype('f8')
    c_od = Start_Time_Finder(np.array(binned_repeat_frame.loc['OD',:]))
    c_orien = Start_Time_Finder(np.array(binned_repeat_frame.loc['Orien',:]))
    c_color = Start_Time_Finder(np.array(binned_repeat_frame.loc['Color',:]))
    # then save each network's repeat by tuning.
    for j,c_start in enumerate(c_od):
        od_orien = c_orien-c_start
        od_color = c_color-c_start
        c_od_orien_waittime = od_orien[od_orien>0]
        c_od_color_waittime = od_color[od_color>0]
        if len(c_od_orien_waittime)>0:
            all_network_waittime.loc[len(all_network_waittime),:] = [cloc_name,'OD','Orien',c_od_orien_waittime.min(),c_start]
        if len(c_od_color_waittime)>0:
            all_network_waittime.loc[len(all_network_waittime),:] = [cloc_name,'OD','Color',c_od_color_waittime.min(),c_start]
        # and save OD's wait time disp too.
        if j < len(c_od)-1:
            all_network_waittime.loc[len(all_network_waittime),:] = [cloc_name,'OD','OD',c_od[j+1]-c_od[j],c_start]

    for j,c_start in enumerate(c_orien):
        orien_od = c_od-c_start
        orien_color = c_color-c_start
        c_orien_od_waittime = orien_od[orien_od>0]
        c_orien_color_waittime = orien_color[orien_color>0]
        if len(c_orien_od_waittime)>0:# to avoid last situation.
            all_network_waittime.loc[len(all_network_waittime),:] = [cloc_name,'Orien','OD',c_orien_od_waittime.min(),c_start]
        if len(c_orien_color_waittime)>0:
            all_network_waittime.loc[len(all_network_waittime),:] = [cloc_name,'Orien','Color',c_orien_color_waittime.min(),c_start]
        if j < len(c_orien)-1:
            all_network_waittime.loc[len(all_network_waittime),:] = [cloc_name,'Orien','Orien',c_orien[j+1]-c_orien[j],c_start]

    for j,c_start in enumerate(c_color):
        color_od = c_od-c_start
        color_orien = c_orien-c_start
        c_color_od_waittime = color_od[color_od>0]
        c_color_orien_waittime = color_orien[color_orien>0]
        if len(c_color_od_waittime)>0:# to avoid last situation.
            all_network_waittime.loc[len(all_network_waittime),:] = [cloc_name,'Color','OD',c_color_od_waittime.min(),c_start]
        if len(c_color_orien_waittime)>0:
            all_network_waittime.loc[len(all_network_waittime),:] = [cloc_name,'Color','Orien',c_color_orien_waittime.min(),c_start]
        if j < len(c_color)-1:
            all_network_waittime.loc[len(all_network_waittime),:] = [cloc_name,'Color','Color',c_color[j+1]-c_color[j],c_start]

ot.Save_Variable(wp,'All_Network_Waittime',all_network_waittime)
#%% ######################### PLOT 3*3 Graphs ############################
from sklearn.metrics import r2_score
all_network_waittime_all_animal = ot.Load_Variable(wp,'All_Network_Waittime.pkl')
all_network_waittime_all_animal['Animal'] = 'Undefined'
all_locname = list(set(all_network_waittime_all_animal['Loc']))
all_locname.sort()
L76_locs = all_locname[:4]
L85_locs = all_locname[4:6]
L91_locs = all_locname[6:]
for i in tqdm(range(len(all_network_waittime_all_animal))):
    c_loc = all_network_waittime_all_animal.loc[i,'Loc']
    if c_loc in L76_locs:
        all_network_waittime_all_animal.loc[i,'Animal'] = 'L76'
    elif c_loc in L85_locs:
        all_network_waittime_all_animal.loc[i,'Animal'] = 'L85'
    elif c_loc in L91_locs:
        all_network_waittime_all_animal.loc[i,'Animal'] = 'L91'

#%% all start times
all_network_waittime = all_network_waittime_all_animal
OD_starts = all_network_waittime['Net_Before'] == 'OD'
Orien_starts = all_network_waittime['Net_Before'] == 'Orien'
Color_starts = all_network_waittime['Net_Before'] == 'Color'
# all end times 
OD_ends = all_network_waittime['Net_After'] == 'OD'
Orien_ends = all_network_waittime['Net_After'] == 'Orien'
Color_ends = all_network_waittime['Net_After'] == 'Color'

# result = all_network_waittime.loc[OD_starts&OD_ends,'Waittime']

def Weibul_Fit_Plotter(ax,disp,x_max):
    #fit
    params = stats.exponweib.fit(disp,floc = 0)
    # params = stats.weibull_min.fit(disp,floc = 0)
    x = np.linspace(0, x_max, 200)
    pdf_fitted = stats.exponweib.pdf(x, *params)
    # plot
    ax.hist(disp, bins=20, density=True, alpha=1,range=[0, x_max])
    ax.plot(x, pdf_fitted, 'r-', label='Fitted')
    ax.set_xlim(0,x_max)

    # calculate r2 at last,using QQ Plot method
    _,(slope, intercept, r) = stats.probplot(disp, dist=stats.exponweib,sparams = params,plot=None, rvalue=True)
    r2 = r**2
    return ax,params,r2

waittime_mean_frame = pd.DataFrame(columns = ['Start_Net','End_Net','Mean','std','R2'])

plt.clf()
plt.cla()
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,7),dpi = 180, sharex='col',sharey='row')
network_seq = ['OD','Orien','Color']
vmax = [100,75,80]

for i,c_start in enumerate([OD_starts,Orien_starts,Color_starts]):
    for j,c_end in enumerate([OD_ends,Orien_ends,Color_ends]):
        c_waittime_disp = all_network_waittime.loc[c_start&c_end,'Waittime']
        # c_waittime_disp = all_network_waittime.loc[c_start&c_end,:]
        axes[i,j],_,c_r2 = Weibul_Fit_Plotter(axes[i,j],np.array(c_waittime_disp).astype('f8'),vmax[j])
        # axes[i,j].hist(c_waittime_disp.groupby('Animal').get_group('L76')['Waittime'],bins=20, density=True, alpha=0.7,range=[0, vmax[j]],label = 'L76')
        # axes[i,j].hist(c_waittime_disp.groupby('Animal').get_group('L85')['Waittime'],bins=20, density=True, alpha=0.7,range=[0, vmax[j]],label = 'L85')
        # axes[i,j].hist(c_waittime_disp.groupby('Animal').get_group('L91')['Waittime'],bins=20, density=True, alpha=0.7,range=[0, vmax[j]],label = 'L91')
        # waittime_mean_frame.loc[len(waittime_mean_frame),:] = [network_seq[i],network_seq[j],c_waittime_disp.mean(),c_waittime_disp.std(),c_r2]
        axes[i,j].text(vmax[j]*0.6,0.075,f'R2 = {c_r2:.3f}')
        axes[i,j].text(vmax[j]*0.6,0.06,f'n = {len(c_waittime_disp)}')
        # L76_meanwaittime = c_waittime_disp.groupby('Animal').get_group('L76')['Waittime'].mean()
        # L85_meanwaittime = c_waittime_disp.groupby('Animal').get_group('L85')['Waittime'].mean()
        # L91_meanwaittime = c_waittime_disp.groupby('Animal').get_group('L91')['Waittime'].mean()
        # axes[i,j].text(vmax[j]*0.4,0.045,f'L76 Mean={L76_meanwaittime:.4f}')
        # axes[i,j].text(vmax[j]*0.4,0.06,f'L85 Mean={L85_meanwaittime:.4f}')
        # axes[i,j].text(vmax[j]*0.4,0.03,f'L91 Mean={L91_meanwaittime:.4f}')


fig.suptitle('Network Waittime',size = 18,y = 0.97)
axes[0,2].legend()
axes[2,0].set_xlabel('To OD',size = 12)
axes[2,1].set_xlabel('To Orientation',size = 12)
axes[2,2].set_xlabel('To Color',size = 12)
axes[0,0].set_ylabel('From OD',size = 12,rotation = 90)
axes[1,0].set_ylabel('From Orientation',size = 12)
axes[2,0].set_ylabel('From Color',size = 12)

fig.tight_layout()







