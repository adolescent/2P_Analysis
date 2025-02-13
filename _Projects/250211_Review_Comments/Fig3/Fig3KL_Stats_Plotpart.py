'''
Only a plot part of stats for repeat freq and repeat similarity.
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

wp = r'D:\_GoogleDrive_Files\#Figs\#250211_Revision1\Fig3'

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
Fig 3L,Repeat Similarity. 
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
fig.savefig(ot.join(wp,'Fig3L_Similarity.png'),bbox_inches='tight')

#%%
'''
Fig 3K,Repeat Freq. 
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
ax.set_xlim(0.06,0.24)
ax.set_ylim(0.04,0.12)

ax.set_yticks([0,0.04,0.08,0.12])
ax.set_xticks([0.05,0.10,0.15,0.20,0.25])
ax.set_yticklabels([0,0.04,0.08,0.12],fontsize = 14)
ax.set_xticklabels([0.05,0.10,0.15,0.20,0.25],fontsize = 14)

plt.show()
fig.savefig(ot.join(wp,'Fig3K_Freq.png'),bbox_inches='tight')
