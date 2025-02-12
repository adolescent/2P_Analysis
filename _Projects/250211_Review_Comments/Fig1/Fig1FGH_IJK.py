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
from Filters import Signal_Filter_v2
import warnings

warnings.filterwarnings("ignore")

expt_folder = r'D:\#Fig_Data\_All_Spon_Data_V1\L76_18M_220902'
savepath = r'D:\_GoogleDrive_Files\#Figs\#250211_Revision1\Fig1'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
sponrun = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
start = sponrun.index[0]
end = sponrun.index[-1]

# generate new spon series, remove LP filter.
# NOTE different shape!
spon_series = Z_refilter(ac,start,end).T

#%%
'''
Fig 1D/E/F, we generate original Heatmaps, without annotating.
'''

# get spon,stim,shuffle frames.
orien_series = Z_refilter(ac,start,end,ac.orienrun).T
spon_shuffle = Spon_Shuffler(spon_series,method='phase',filter_para=(0.005,0.65))
# transfer them into pd frame, for further process.
spon_series = pd.DataFrame(spon_series,columns = ac.acn,index = range(len(spon_series)))
orien_series = pd.DataFrame(orien_series,columns = ac.acn,index = range(len(orien_series)))
spon_shuffle_frame = pd.DataFrame(spon_shuffle,columns = ac.acn,index = range(len(spon_shuffle)))

# Sort Orien By Cells actually we sort only by raw data.
rank_index = pd.DataFrame(index = ac.acn,columns=['Best_Orien','Sort_Index','Sort_Index2'])
for i,cc in enumerate(ac.acn):
    rank_index.loc[cc]['Best_Orien'] = ac.all_cell_tunings[cc]['Best_Orien']
    if ac.all_cell_tunings[cc]['Best_Orien'] == 'False':
        rank_index.loc[cc]['Sort_Index']=-1
        rank_index.loc[cc]['Sort_Index2']=0
    else:
        orien_tunings = float(ac.all_cell_tunings[cc]['Best_Orien'][5:])
        # rank_index.loc[cc]['Sort_Index'] = np.sin(np.deg2rad(orien_tunings))
        rank_index.loc[cc]['Sort_Index'] = orien_tunings
        rank_index.loc[cc]['Sort_Index2'] = np.cos(np.deg2rad(orien_tunings))
# actually we sort only by raw data.
sorted_cell_sequence = rank_index.sort_values(by=['Sort_Index'],ascending=False)
# and we try to reindex data.
sorted_stim_response = orien_series.T.reindex(sorted_cell_sequence.index).T
sorted_spon_response = spon_series.T.reindex(sorted_cell_sequence.index).T
sorted_shuffle_response = spon_shuffle_frame.T.reindex(sorted_cell_sequence.index).T
#%% Plot Cell Stim Maps
plt.clf()
plt.cla()

# cbar_ax = fig.add_axes([1, .35, .01, .3])
# label_size = 14
# title_size = 18
vmax = 4
vmin = -2

