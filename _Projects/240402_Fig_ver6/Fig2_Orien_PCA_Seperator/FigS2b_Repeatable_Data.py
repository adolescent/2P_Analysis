'''
This script will compar data from 2 day's spontaneous response.


We hope we can get similar response infos.

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
from Classifier_Analyzer import *

import warnings
warnings.filterwarnings("ignore")

wp1 = r'D:\_All_Spon_Data_V1\L76_15A_220812'
ac1 = ot.Load_Variable(wp1,'Cell_Class.pkl')
spon_series1 = ot.Load_Variable(wp1,'Spon_Before.pkl')

wp2 = r'D:\_All_Spon_Data_V1_Repeat\L76_15A_220721'


#%% #################STEP0, GENERATE REPEAT SPON TRAINS.############################
############################### DO NOT RERUN FREQUENTLY.####################
ac2 = Stim_Cells(wp2,od = 6,orien = 7,color = 8)
# ac2.Calculate_Frame_Labels()
ac2.Calculate_All()
ac2.Plot_T_Graphs()
ot.Save_Variable(wp2,'Cell_Class',ac2)
repeat_spon_series = ac2.Z_Frames['1-001']
# get 100 windowed std graphs.
step = 100
winnum  = (len(repeat_spon_series)-100)//step
std_matrix = np.zeros(shape = (winnum,repeat_spon_series.shape[1]))
for i in range(winnum):
    c_series = np.array(repeat_spon_series)[i*step:100+i*step]
    c_std = c_series.std(0)
    std_matrix[i,:] = np.array(c_std)
plt.plot(std_matrix.mean(1))
# we find that we can use spon from 4600.
used_spon = repeat_spon_series.loc[4600:,:]
ot.Save_Variable(wp2,'Spon_Before',used_spon)
spon_series2 = used_spon
#%%################## Step 1- Get 2 repeat compares.#################
pcnum = 10

_,_,spon_models1 = Z_PCA(Z_frame=spon_series1,sample='Frame',pcnum=pcnum)
model_var_ratio1 = np.array(spon_models1.explained_variance_ratio_)
print(f'{pcnum} PCs explain Spontaneous VAR {model_var_ratio1[:pcnum].sum()*100:.1f}%')

_,_,spon_models2 = Z_PCA(Z_frame=spon_series2,sample='Frame',pcnum=pcnum)
model_var_ratio2 = np.array(spon_models2.explained_variance_ratio_)
print(f'{pcnum} PCs explain Spontaneous VAR {model_var_ratio2[:pcnum].sum()*100:.1f}%')

#%% Plot PC explained variance here.
plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=2,figsize = (8,4),dpi = 144,sharey= True)
sns.barplot(y = model_var_ratio1*100,x = np.arange(1,11),ax = ax[0])
sns.barplot(y = model_var_ratio2*100,x = np.arange(1,11),ax = ax[1])

ax[0].set_ylabel('Explained Variance (%)',size = 12)
ax[0].set_xlabel('PC',size = 12)
ax[1].set_xlabel('PC',size = 12)
ax[0].set_title('0721 PC explained Variance',size = 14)
ax[1].set_title('0812 PC explained Variance',size = 14)
fig.tight_layout()
#%% Compare orientation seperation and recovered maps.

analyzer1 = Classify_Analyzer(ac = ac1,umap_model=spon_models1,spon_frame=spon_series1,od = 0,orien = 1,color = 0,isi = True)
analyzer1.Train_SVM_Classifier(C=1)
spon_label1 = analyzer1.spon_label

analyzer2 = Classify_Analyzer(ac = ac2,umap_model=spon_models2,spon_frame=spon_series2,od = 0,orien = 1,color = 0,isi = True)
analyzer2.Train_SVM_Classifier(C=1)
spon_label2 = analyzer2.spon_label

#%% Plot compared orientation repeat frequency.
All_Orientation_Repeats = pd.DataFrame(0.0,index = [0,1],columns = ['Repeat','Freq','Prop'])
event_count1 = Event_Counter(spon_label1>0)
event_count2 = Event_Counter(spon_label2>0)

All_Orientation_Repeats.loc[0,:] = ['0812',event_count1*1.301/len(spon_label1),np.sum(spon_label1>0)/len(spon_label1)]
All_Orientation_Repeats.loc[1,:] = ['0721',event_count2*1.301/len(spon_label2),np.sum(spon_label2>0)/len(spon_label2)]

plt.clf()
plt.cla()
plotable_data = All_Orientation_Repeats.melt(id_vars='Repeat')
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (3,5),dpi = 144,sharey= True)
sns.barplot(data = plotable_data,y = 'value',x = 'variable',hue = 'Repeat',ax = ax)
ax.set_xticklabels(['Freq(Hz)','Prop'])
ax.set_title('Orientation Ensemble in 2 repeats')
fig.tight_layout()

#%% Plot recovered graphs.

analyzer1.Get_Stim_Spon_Compare(od = False,color = False)
stim_graphs1 = analyzer1.stim_recover
spon_graphs1 = analyzer1.spon_recover
graph_lists = ['Orien0','Orien45','Orien90','Orien135']
analyzer1.Similarity_Compare_Average(od = False,color = False)
all_corr1 = analyzer1.Avr_Similarity

analyzer2.Get_Stim_Spon_Compare(od = False,color = False)
stim_graphs2 = analyzer2.stim_recover
spon_graphs2 = analyzer2.spon_recover
graph_lists = ['Orien0','Orien45','Orien90','Orien135']
analyzer2.Similarity_Compare_Average(od = False,color = False)
all_corr2 = analyzer2.Avr_Similarity

#%%
plt.clf()
plt.cla()
value_max = 2
value_min = -1
font_size = 16
fig,axes = plt.subplots(nrows=2, ncols=4,figsize = (14,7),dpi = 180)
cbar_ax = fig.add_axes([.92, .45, .01, .2])

for i,c_map in enumerate(graph_lists):
    sns.heatmap(stim_graphs2[c_map][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[0,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True)
    sns.heatmap(spon_graphs2[c_map][1],center = 0,xticklabels=False,yticklabels=False,ax = axes[1,i],vmax = value_max,vmin = value_min,cbar_ax= cbar_ax,square=True,cbar_kws={'label': 'Z Scored Activity'})
    axes[0,i].set_title(c_map,size = font_size)

axes[0,0].set_ylabel('Stimulus',rotation=90,size = font_size)
axes[1,0].set_ylabel('Spontaneous',rotation=90,size = font_size)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
dist = 0.195
height = 0.485
plt.figtext(0.18, height, f'R2 = {all_corr2.iloc[0,0]:.3f}',size = 14)
plt.figtext(0.18+dist, height, f'R2 = {all_corr2.iloc[2,0]:.3f}',size = 14)
plt.figtext(0.18+dist*2, height, f'R2 = {all_corr2.iloc[4,0]:.3f}',size = 14)
plt.figtext(0.18+dist*3, height, f'R2 = {all_corr2.iloc[6,0]:.3f}',size = 14)
cbar_ax.yaxis.label.set_size(12)
# fig.tight_layout()


plt.show()