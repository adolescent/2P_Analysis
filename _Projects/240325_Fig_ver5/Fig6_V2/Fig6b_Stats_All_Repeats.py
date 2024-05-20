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
from Cell_Class.Timecourse_Analyzer import *



all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V2'))
# all_path_dic.pop(4)
# all_path_dic.pop(6)

import warnings
warnings.filterwarnings("ignore")
all_repeat_similarity = ot.Load_Variable(r'D:\_Path_For_Figs\240228_Figs_v4\Fig6\VerA_Direct_Spon_PCA','Orien_Repeat_Similarity.pkl')
all_repeat_freq = ot.Load_Variable(r'D:\_Path_For_Figs\240228_Figs_v4\Fig6\VerA_Direct_Spon_PCA','Orien_Repeat_Freq.pkl')

#%% ###############################FIG 6C , Plot repeat similarity #############################
plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (2,5),dpi = 180)
ax.axhline(y = 0,color='gray', linestyle='--')
sns.boxplot(data = all_repeat_similarity,x = 'Data_Type',y = 'Corr',hue = 'Network',ax = ax,showfliers = False)
ax.set_title('Stim-like Ensemble Repeat Similarity',size = 10)
ax.set_xlabel('')
ax.set_ylabel('Pearson R')
ax.legend(title = 'Network',fontsize = 5)
ax.set_xticklabels(['Real Data','Random Select'],size = 7)
plt.show()

#%% ###############################FIG 6D , Plot repeat Frequency #############################
plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (2,5),dpi = 180)
ax.axhline(y = 0,color='gray', linestyle='--')

# sns.boxplot(data = all_repeat_freq,x = 'Loc',y = 'Freq',hue = 'Data_Type',ax = ax,showfliers = 0,legend = True)
sns.boxplot(data = all_repeat_freq,x = 'Network',y = 'Freq',hue = 'Data_Type',ax = ax,showfliers = 0,legend = True,hue_order=['Real_Data','Phase_Shuffle'],width=0.5)
ax.set_title('Stim-like Ensemble Repeat Similarity',size = 10)
ax.set_xlabel('')
ax.set_ylabel('Frequency(Hz)')
ax.legend(title = 'Network',fontsize = 5)
# ax.set_xticklabels(['Real Data','Random Select'],size = 7)
# ax.set_xticklabels([''],size = 7)
plt.show()


#%%############################ BELOW IS AN UNITE GRAPH, MIGHT BE USEFUL


plt.clf()
plt.cla()
# set graph
fig,axes = plt.subplots(nrows=1, ncols=2,figsize = (10,7),dpi = 180)
axes[0].axhline(y = 0,color='gray', linestyle='--')
axes[1].axhline(y = 0,color='gray', linestyle='--')
fig.suptitle('Orientation-Like Ensemble',size = 20,y = 0.98)

## plot similarity
sns.boxplot(data = all_repeat_similarity,x = 'Data_Type',y = 'Corr',hue = 'Network',ax = axes[0],showfliers = False,width = 0.5)
axes[0].set_title('Repeat Similarity',size = 16)
axes[0].set_xlabel('')
axes[0].set_ylabel('Pearson R',size = 14)
axes[0].legend(fontsize = 10)
# axes[0].legend(['Orien 0', 'Orien 45','Orien 90','Orien 135'],prop = { "size": 10 })
axes[0].set_xticklabels(['Real Data','Random Select'],size = 14)
axes[0].set_ylim(-0.3,0.9)

## Plot frequency
used_freq = all_repeat_freq[all_repeat_freq['Data_Type']!= 'Dim_Shuffle']
sns.boxplot(data = used_freq,x = 'Data_Type',y = 'Freq',hue = 'Data_Type',ax = axes[1],showfliers = 0,legend = True,hue_order=['Real_Data','Phase_Shuffle'],width=0.25)

axes[1].set_title('Repeat Frequency',size = 16)
axes[1].set_xlabel('')
axes[1].set_xticklabels(['Real Data','Shuffled Data'],size = 14)
axes[1].set_ylabel('Frequency(Hz)',size = 14)
axes[1].yaxis.tick_right()
axes[1].yaxis.set_label_position("right")
axes[1].legend([])
