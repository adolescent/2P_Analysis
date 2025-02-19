'''
This compares network recover similarity between V1 and V2.

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
from Review_Fix_Funcs import *

import warnings
warnings.filterwarnings("ignore")

wp = r'D:\_GoogleDrive_Files\#Figs\#250211_Revision1\Fig3'
wp_v2 = r'D:\_GoogleDrive_Files\#Figs\#250211_Revision1\Fig5'

#%% load in v1 and v2 similarities.
orien_similar = ot.Load_Variable(wp,'Orien_Repeat_Similarity.pkl')
hue_similar = ot.Load_Variable(wp,'Hue_Repeat_Similarity.pkl')
orien_similar_v2 = ot.Load_Variable(wp_v2,'Orien_Repeat_Similarity_V2.pkl')
hue_similar_v2 = ot.Load_Variable(wp_v2,'Color_Repeat_Similarity_V2.pkl')
# hue_similar_v2 = hue_similar_v2.groupby('Loc').get_group('L85_6B_220825')

#%%
'''
Fig 5K, Compare V1 and V2 Orien, Color's similarity.
'''
# calculate part
# get concated matrix first.
hue_similar['Brain Area'] = 'V1'
hue_similar_v2['Brain Area'] = 'V2'
hue_similar_v2.columns = ['Loc','Network','Corr','Map_Type','Data_Type','Brain Area']
orien_similar['Brain Area'] = 'V1'
orien_similar_v2['Brain Area'] = 'V2'
# orien_similar_v2.columns = ['Corr','Network','Data_Type','Map_Type','Brain Area']
# concat V1 and V2 data
all_hue_similar = pd.concat([hue_similar,hue_similar_v2],ignore_index = True)
all_orien_similar = pd.concat([orien_similar,orien_similar_v2],ignore_index = True)

# select only real data.
all_hue_similar = all_hue_similar[all_hue_similar['Data_Type']=='Real Data']
all_orien_similar = all_orien_similar[all_orien_similar['Data_Type']=='Real Data']
#%% Plot part

plotable = pd.concat([all_hue_similar,all_orien_similar],ignore_index = True)

plt.clf()
plt.cla()
fontsize = 12
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (2,3.5),dpi = 300)
sns.barplot(data = plotable,x = 'Map_Type',y = 'Corr',ax = ax,hue = 'Brain Area',capsize = 0.2,hue_order = ['V1','V2'],width = 0.5)

ax.legend(fontsize=fontsize)
ax.set_ylim(0,1)
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize = fontsize)
ax.set_xticklabels(['Color','Orien'],fontsize = fontsize)
# ax.legend(['Real PC', 'Shuffled PC'],prop = { "size": fontsize })
fig.savefig(ot.join(wp_v2,'Fig5M_V1_V2_Compare.png'),bbox_inches='tight')


#%% print welch test's result here. It might not be so reliable.
v1_hues = all_hue_similar[all_hue_similar['Brain Area']=='V1']['Corr'].astype('f8')
v2_hues = all_hue_similar[all_hue_similar['Brain Area']=='V2']['Corr'].astype('f8')
v1_oriens = all_orien_similar[all_orien_similar['Brain Area']=='V1']['Corr'].astype('f8')
v2_oriens = all_orien_similar[all_orien_similar['Brain Area']=='V2']['Corr'].astype('f8')

hue_r,hue_p = stats.ttest_ind(np.array(v1_hues),np.array(v2_hues))
orien_r,orien_p = stats.ttest_ind(np.array(v1_oriens),np.array(v2_oriens))