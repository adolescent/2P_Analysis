'''

This script will calculate auto correlation between 3 repeat frames.

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
from sklearn.model_selection import cross_val_score
from sklearn import svm
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import colorsys
import matplotlib as mpl

work_path = r'D:\_Path_For_Figs\230507_Figs_v7\Cell_PCA_oriens'
all_path_dic = list(ot.Get_Subfolders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)

#%%
'''
Part 1, read in all repeat trains.
'''
all_repeat_trains = {}
for i,cloc in enumerate(all_path_dic):
    cloc_name = cloc.split('\\')[-1]
    all_repeat_trains[cloc_name] = ot.Load_Variable(cloc,'All_Spon_Repeats_PCA10.pkl')

all_locs = list(all_repeat_trains.keys())
#%% Get auto correlation plot
# a = np.array(all_repeat_trains[all_locs[0]]['OD']>0)
max_lag = 50
# corrs = np.zeros(shape = (max_lag,8))
all_auto_corr = pd.DataFrame(columns = ['Loc','Lag','Network','Corr'])

for j in range(8):
    cloc = all_locs[j]
    od = np.array(all_repeat_trains[all_locs[j]]['OD']>0)
    color = np.array(all_repeat_trains[all_locs[j]]['Color']>0)
    orien = np.array(all_repeat_trains[all_locs[j]]['Orien']>0)
    
    all_trains = [od,orien,color]
    # corrs = np.zeros(max_lag*2)
    # cen_part = a[max_lag:-max_lag]
    for k,c_net in enumerate(['OD','Orien','Color']):
        a = all_trains[k]
        cen_part = a[:-max_lag]
        for i in range(max_lag):
            corr_parts = a[i:i+len(cen_part)]
            c_r,_ = stats.pearsonr(cen_part,corr_parts)
            # corrs[i,j] = c_r
            all_auto_corr.loc[len(all_auto_corr),:] = [cloc,i,c_net,c_r]
all_auto_corr['Lag'] = all_auto_corr['Lag'].astype('f8')
all_auto_corr['Corr'] = all_auto_corr['Corr'].astype('f8')
# plt.plot(corrs.mean(1))
#%%

plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (5,7),dpi = 180)
ax.axvline(x = 6,color = 'gray',linestyle = '--')
sns.lineplot(data = all_auto_corr,x = 'Lag',y = 'Corr',hue = 'Network',ax = ax)
ax.set_ylim(-0.1,0.4)