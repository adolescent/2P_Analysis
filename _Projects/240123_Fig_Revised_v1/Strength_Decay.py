'''
This part will give a ensemble size or activation strength drop of all location datas.

This will be used on determine stim on locations.
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


work_path = r'D:\_Path_For_Figs\240123_Graph_Revised_v1\Fig1_Revised'
expt_folder = r'D:\_All_Spon_Data_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
ac.wp = expt_folder
c_spon = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
c_model = ot.Load_Variable(expt_folder,'All_Stim_UMAP_3D_20comp.pkl')
ac.Regenerate_Cell_Graph()
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)

import warnings
warnings.filterwarnings("ignore")
#%%########################## 1. DEPICT AVR STRENGTH EXAMPLE ######################
avr_response_raw = c_spon.mean(1)
decay_response = np.sort(avr_response_raw)[::-1] # decay order.
response_ensemble = c_spon>1
response_ensemble = response_ensemble.sum(1) # this method have some problems..
decay_ensemble = np.sort(response_ensemble)[::-1]
# get all model classified id.
analyzer = UMAP_Analyzer(ac = ac,umap_model = c_model,spon_frame = c_spon)
analyzer.Train_SVM_Classifier()
spon_train = analyzer.spon_label
spon_ons = np.where(spon_train>0)[0]
spon_offs = np.where(spon_train==0)[0]
avr_spon_on = c_spon.iloc[spon_ons,:].mean(1)
decay_on = np.sort(avr_spon_on)[::-1]
avr_spon_off = c_spon.iloc[spon_offs,:].mean(1)
decay_off = np.sort(avr_spon_off)[::-1]
#%% Plot CDF of all spon data and avr 
plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,4),dpi = 180,width_ratios=[2, 1],sharey=False)
ax[0].axvline(x = np.median(avr_spon_on),color = 'gray',linestyle = '--')
ax[0].axvline(x = np.median(avr_spon_off),color = 'gray',linestyle = '--')
ax[0].text(-0.5,0.3,f'Response Diff={(np.median(avr_spon_on)-np.median(avr_spon_off)):.2f}')
# ymax=decay_response[2676]
sns.lineplot(y = np.array(range(len(decay_off)))/len(decay_off),x = decay_off,ax = ax[0])
# add an end point for on frames.
sns.lineplot(y = np.append(np.array(range(len(decay_on)))/len(decay_on),1),x = np.append(decay_on,decay_response.min()),ax = ax[0])
# sns.lineplot(y = np.array(range(len(decay_off)))/len(decay_off),x = decay_off,ax = ax)
# sns.lineplot(y = np.array(range(len(decay_ensemble)))/len(decay_ensemble),x = decay_ensemble,ax = ax)
# ax.set_xlim(524,-20)
ax[0].set_xlim(3,-3)
ax[0].set_title('CDF of Spontaneous Response Frames')
ax[0].set_ylabel('Prob.')
ax[0].set_xlabel('Average Z Score')

# and plot histo graph of spon data.
seperated_response = pd.DataFrame(0,index = range(len(decay_off)),columns = ['Strength','Type'])
seperated_response.loc[:,'Type'] = 'Non-Stim Repeats'
seperated_response.loc[:,'Strength'] = decay_off
seperated_response2 = pd.DataFrame(0,index = range(len(decay_on)),columns = ['Strength','Type'])
seperated_response2.loc[:,'Type'] = 'Spon Repeats'
seperated_response2.loc[:,'Strength'] = decay_on
plotable_data = pd.concat([seperated_response,seperated_response2])
# sns.histplot(x = decay_response,stat="density",alpha = 0.5,ax = ax[1])
# sns.histplot(x = decay_on,stat="density",alpha = 0.5,ax = ax[1])
g = sns.histplot(data=plotable_data,x = 'Strength', stat='probability',hue = 'Type', common_norm=False,ax = ax[1])

sns.move_legend(g, "lower center", bbox_to_anchor=(.75, 1), ncol=1, title=None, frameon=False)
ax[1].yaxis.set_label_position('right')
ax[1].yaxis.tick_right() 
ax[1].set_ylabel('Probability Density')
ax[1].set_xlabel('Average Z Score')
fig.tight_layout()
plt.show()


#%%########################## 2. STRENGTH DECAY ALL POINTS ############################################
all_spon_strength = pd.DataFrame(columns = ['Strength','Type','Loc'])
for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    c_model = ot.Load_Variable(cloc,'All_Stim_UMAP_3D_20comp.pkl')
    c_analyzer = UMAP_Analyzer(ac = ac,umap_model=c_model,spon_frame=c_spon)
    c_analyzer.Train_SVM_Classifier(C=1)
    c_spon_series = c_analyzer.spon_label
    c_spon_response = np.array(c_spon.mean(1))
    for j in range(len(c_spon_series)):
        if c_spon_series[j] == 0:
            all_spon_strength.loc[len(all_spon_strength),:] = [c_spon_response[j],'Unclassified Spontaneous',cloc_name]
        else:
            all_spon_strength.loc[len(all_spon_strength),:] = [c_spon_response[j],'Classified Spontaneous',cloc_name]
    # spon_ons = np.where(c_spon_series>0)[0]
    # spon_offs = np.where(c_spon_series==0)[0]
    # avr_spon_on = c_spon.iloc[spon_ons,:].mean(1)
    # avr_spon_off = c_spon.iloc[spon_offs,:].mean(1)
ot.Save_Variable(work_path,'All_Spon_Strength',all_spon_strength)
#%% Plot 
on_series = np.array(all_spon_strength.groupby('Type').get_group('Classified Spontaneous')['Strength'])
off_series = np.array(all_spon_strength.groupby('Type').get_group('Unclassified Spontaneous')['Strength'])
decay_on = np.sort(on_series)[::-1]
decay_off = np.sort(off_series)[::-1]

plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,4),dpi = 180,width_ratios=[2, 1],sharey=False)
ax[0].axvline(x = np.median(on_series),color = 'gray',linestyle = '--')
ax[0].axvline(x = np.median(off_series),color = 'gray',linestyle = '--')
ax[0].text(-0.4,0.3,f'Response Diff={(np.median(on_series)-np.median(off_series)):.2f}')
# ymax=decay_response[2676]
sns.lineplot(y = np.array(range(len(decay_off)))/len(decay_off),x = decay_off,ax = ax[0])
# add an end point for on frames.
sns.lineplot(y = np.append(np.array(range(len(decay_on)))/len(decay_on),1),x = np.append(decay_on,decay_off.min()),ax = ax[0])
# sns.lineplot(y = np.array(range(len(decay_off)))/len(decay_off),x = decay_off,ax = ax)
# sns.lineplot(y = np.array(range(len(decay_ensemble)))/len(decay_ensemble),x = decay_ensemble,ax = ax)
# ax.set_xlim(524,-20)
ax[0].set_xlim(3,-3)
ax[0].set_title('CDF of Spontaneous Response Frames')
ax[0].set_ylabel('Prob.')
ax[0].set_xlabel('Average Z Score')
g = sns.histplot(data=all_spon_strength,x = 'Strength', stat='probability',hue = 'Type', common_norm=False,ax = ax[1],bins = 50)
sns.move_legend(g, "lower center", bbox_to_anchor=(.75, 1), ncol=1, title=None, frameon=False)
ax[1].yaxis.set_label_position('right')
ax[1].yaxis.tick_right()
ax[1].set_ylabel('Probability Density')
ax[1].set_xlabel('Average Z Score')
fig.tight_layout()
plt.show()