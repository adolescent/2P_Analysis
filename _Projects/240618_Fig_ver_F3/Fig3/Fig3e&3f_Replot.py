'''

Just change the shape of graph.

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
from sklearn.model_selection import cross_val_score
from sklearn import svm
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
from Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *

import warnings
warnings.filterwarnings("ignore")

wp = r'D:\_Path_For_Figs\240520_Figs_ver_F1\Fig3_PCA_SVM\VARs_From_v4'


#%%
'''
Step0, we generate all network's similarity and repeat frequency. For easy we use pre generated data, but we can easily do it again.
'''
orien_similar = ot.Load_Variable(wp,'Orien_Repeat_Similarity.pkl')
orien_freq = ot.Load_Variable(wp,'Orien_Repeat_Freq.pkl')
od_similar = ot.Load_Variable(wp,'OD_Repeat_Similarity.pkl')
od_freq = ot.Load_Variable(wp,'OD_Repeat_Freq.pkl')
hue_similar = ot.Load_Variable(wp,'Hue_Repeat_Similarity.pkl')
hue_freq = ot.Load_Variable(wp,'Hue_Repeat_Freq.pkl')


#%%
'''
Fig 3E, here we only plot radom select and real data's similarity in OD and orientation.

This time we add color into it.
'''
# all_similar = pd.concat([orien_similar,od_similar,hue_similar], ignore_index=True)
all_similar = pd.concat([orien_similar,od_similar], ignore_index=True)

plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (3,5),dpi = 180)
ax.axhline(y = 0,color='gray', linestyle='--')
sns.barplot(data = all_similar,x = 'Data_Type',y = 'Corr',hue = 'Map_Type',ax = ax,width = 0.6,capsize=.2,err_kws={"linewidth": 1})
# ax.set_title('Stim-like Ensemble Repeat Similarity',size = 9)
ax.set_xlabel('')
# ax.set_ylim(-0.2,1)
ax.set_ylabel('Pearson R')
ax.legend(title = 'Network',fontsize = 8)
ax.set_xticklabels(['Real Data','Random Select'],size = 8)
plt.show()

#%%
'''
Fig 3E, Plot Repeat Graph Similarity here.
'''

real_orien_avr = orien_similar[orien_similar['Data_Type']=='Real Data'].groupby('Loc')['Corr'].mean()
real_od_avr = od_similar[od_similar['Data_Type']=='Real Data'].groupby('Loc')['Corr'].mean()
plotable = pd.DataFrame([real_orien_avr,real_od_avr],columns = real_od_avr.index,index = ['Orien','OD']).T

plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (4,3),dpi = 300)
sns.scatterplot(data = plotable,x = 'Orien',y = 'OD',ax = ax,hue = 'Loc',legend=False)
ax.set_ylim(0,1)
ax.set_xlim(0,1)
# ax.set_ylabel('OD Similarity')
# ax.set_xlabel('Orientation Similarity')
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_yticks(np.linspace(0,1,5))
ax.set_xticks(np.linspace(0,1,5))
ax.set_yticklabels(np.linspace(0,1,5),fontsize = 14)
ax.set_xticklabels(np.linspace(0,1,5),fontsize = 14)

#%%
'''
Fig 3F, Plot Repeat Graph Event Frequency Here.
'''
real_orien_freq = orien_freq[orien_freq['Data_Type']=='Real_Data'].groupby('Loc')['Freq'].mean()
real_od_freq = od_freq[od_freq['Data_Type']=='Real_Data'].groupby('Loc')['Freq'].mean()
plotable_freq = pd.DataFrame([real_orien_freq,real_od_freq],columns = real_od_avr.index,index = ['Orien','OD']).T

plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (4,3),dpi = 180)
sns.scatterplot(data = plotable_freq,x = 'Orien',y = 'OD',ax = ax,hue = 'Loc',legend=False)
# ax.set_ylabel('OD Freq (Hz)')
# ax.set_xlabel('Orientation Freq (Hz)')
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_xlim(0.06,0.16)
ax.set_ylim(0.02,0.1)

ax.set_yticks([0,0.04,0.08,0.12])
ax.set_xticks([0,0.05,0.10,0.15,0.20])
ax.set_yticklabels([0,0.04,0.08,0.12],fontsize = 14)
ax.set_xticklabels([0,0.05,0.10,0.15,0.20],fontsize = 14)

plt.show()
