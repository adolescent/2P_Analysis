'''
Stats all network repeat frequency, and repeat similarity.

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
from Cell_Class.Timecourse_Analyzer import *

import warnings
warnings.filterwarnings("ignore")

wp = r'D:\_Path_For_Figs\240228_Figs_v4\Fig2\VerA_Direct_Spon_PCA'
orien_similar = ot.Load_Variable(wp,'Orien_Repeat_Similarity.pkl')
orien_freq = ot.Load_Variable(wp,'Orien_Repeat_Freq.pkl')
od_similar = ot.Load_Variable(wp,'OD_Repeat_Similarity.pkl')
od_freq = ot.Load_Variable(wp,'OD_Repeat_Freq.pkl')
hue_similar = ot.Load_Variable(wp,'Hue_Repeat_Similarity.pkl')
hue_freq = ot.Load_Variable(wp,'Hue_Repeat_Freq.pkl')

#%% ########################### FIG 4E-All NETWORK SIMILARIY #################################
all_similar = pd.concat([orien_similar,od_similar,hue_similar], ignore_index=True)

plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (2.5,5),dpi = 180)
ax.axhline(y = 0,color='gray', linestyle='--')
sns.boxplot(data = all_similar,x = 'Data_Type',y = 'Corr',hue = 'Map_Type',ax = ax,showfliers = False,width = 0.5)
ax.set_title('Stim-like Ensemble Repeat Similarity',size = 9)
ax.set_xlabel('')
# ax.set_ylim(-0.2,1)
ax.set_ylabel('Pearson R')
ax.legend(title = 'Network',fontsize = 8)
ax.set_xticklabels(['Real Data','Random Select'],size = 8)
plt.show()

#%% ########################### FIG 4F-All NETWORK REPEAT FREQUENCY #################################
all_freq = pd.concat([orien_freq,od_freq,hue_freq], ignore_index=True)
all_freq = all_freq[all_freq['Data_Type'] != 'Dim_Shuffle']
all_freq['Network'] = all_freq['Network'].replace('Hue', 'Color')

plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (2.5,5),dpi = 180)
ax.axhline(y = 0,color='gray', linestyle='--')

# sns.boxplot(data = all_repeat_freq,x = 'Loc',y = 'Freq',hue = 'Data_Type',ax = ax,showfliers = 0,legend = True)
sns.boxplot(data = all_freq,x = 'Data_Type',y = 'Freq',hue = 'Network',ax = ax,showfliers = 0,legend = True,hue_order=['Orien','OD','Color'],width=0.5)
ax.set_title('Stim-like Ensemble Repeat Frequency',size = 10)
ax.set_xlabel('')
ax.set_ylabel('Frequency(Hz)')
ax.legend(title = 'Network',fontsize = 8)
ax.set_xticklabels(['Real Data','Phase Shuffle'],size = 8)


# ax.set_xticklabels([''],size = 7)
plt.show()
