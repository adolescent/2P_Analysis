'''
This script will compare orientation and color repeat's similarity between V1 and V2.

Remember, for color, only 1 point is possible.
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
# orien_freq = ot.Load_Variable(wp,'Orien_Repeat_Freq.pkl')
# od_similar = ot.Load_Variable(wp,'OD_Repeat_Similarity.pkl')
# od_freq = ot.Load_Variable(wp,'OD_Repeat_Freq.pkl')
hue_similar = ot.Load_Variable(wp,'Hue_Repeat_Similarity.pkl')
# hue_freq = ot.Load_Variable(wp,'Hue_Repeat_Freq.pkl')
orien_similar_v2 = ot.Load_Variable(r'D:\_Path_For_Figs\240228_Figs_v4\Fig6\VerA_Direct_Spon_PCA','Orien_Repeat_Similarity.pkl')

#%% and we need to calculate L85's color similarity.
hue_loc = r'D:\_All_Spon_Data_V2\L85_6B_220825'
hue_ac = ot.Load_Variable(hue_loc,'Cell_Class.pkl')
hue_spon = np.array(ot.Load_Variable(hue_loc,'Spon_Before.pkl'))
pcnum = 10

comp,coords,model = Z_PCA(hue_spon,'Frame',pcnum)
analyzer = Classify_Analyzer(ac = hue_ac,model = model,spon_frame=hue_spon,od = False,orien = False)
analyzer.Similarity_Compare_Average()
hue_similar_v2 = analyzer.Avr_Similarity
#%%
'''
Fig 5E, Compare V1 and V2 Orien, Color's similarity.
'''
# get concated matrix first.
hue_similar['Brain Area'] = 'V1'
hue_similar_v2['Brain Area'] = 'V2'
hue_similar_v2.columns = ['Corr','Network','Data_Type','Map_Type','Brain Area']
orien_similar['Brain Area'] = 'V1'
orien_similar_v2['Brain Area'] = 'V2'
# orien_similar_v2.columns = ['Corr','Network','Data_Type','Map_Type','Brain Area']
# concat V1 and V2 data
all_hue_similar = pd.concat([hue_similar,hue_similar_v2],ignore_index = True)
all_orien_similar = pd.concat([orien_similar,orien_similar_v2],ignore_index = True)

# select only real data.
all_hue_similar = all_hue_similar[all_hue_similar['Data_Type']=='Real Data']
all_orien_similar = all_orien_similar[all_orien_similar['Data_Type']=='Real Data']
#%% Plot here, x as network, hue as brain area.

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

#%% print welch test's result here. It might not be so reliable.
v1_hues = all_hue_similar[all_hue_similar['Brain Area']=='V1']['Corr'].astype('f8')
v2_hues = all_hue_similar[all_hue_similar['Brain Area']=='V2']['Corr'].astype('f8')
v1_oriens = all_orien_similar[all_orien_similar['Brain Area']=='V1']['Corr'].astype('f8')
v2_oriens = all_orien_similar[all_orien_similar['Brain Area']=='V2']['Corr'].astype('f8')

hue_r,hue_p = stats.ttest_ind(np.array(v1_hues),np.array(v2_hues))
orien_r,orien_p = stats.ttest_ind(np.array(v1_oriens),np.array(v2_oriens))