'''
This script will calculate and show all cells tuning in V1.
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
from Classifier_Analyzer import *

import warnings
warnings.filterwarnings("ignore")

# wp = r'D:\_All_Spon_Data_V1\L76_18M_220902'
# ac = ot.Load_Variable(wp,'Cell_Class.pkl')
# spon_series = ot.Load_Variable(wp,'Spon_Before.pkl')
savepath = r'D:\_GoogleDrive_Files\#Figs\Comments240618_Figs_ver_F3\240618_Figs_ver_F3\Fig1_Brief'

all_path_dic = list(ot.Get_Subfolders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)

#%%
'''
Step0, calculate all locations tuning. Cell Class have method to do it.
'''

for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable(cloc,'Cell_Class.pkl')
    c_tuning = ac.all_cell_tunings
    c_od_tuning = c_tuning.loc['OD_index',:]
    c_orien_tuning = c_tuning.loc['Best_Orien',:]
    c_color_tuning = c_tuning.loc['Best_Color',:]
    c_tuning_frame = pd.DataFrame([c_od_tuning,c_orien_tuning,c_color_tuning],index = ['OD','Orien','Color']).T
    c_tuning_frame['Loc'] = cloc_name
    if i == 0:
        all_tuning_frame = copy.deepcopy(c_tuning_frame)
    else:
        all_tuning_frame = pd.concat([all_tuning_frame,c_tuning_frame],ignore_index=True)

ot.Save_Variable(savepath,'All_Cell_Tunings',all_tuning_frame)
#%%
'''
Fig S1c, Plot distribution of all cells tuning.
'''
#%% OD
plt.clf()
plt.cla()
fig, ax = plt.subplots(figsize = (5,4),dpi = 180)
sns.histplot(data = all_tuning_frame,x = 'OD',ax = ax,bins = np.linspace(-1,1,8))
ax.set_xlabel('OD Index')

#%% orientation
plotable = copy.deepcopy(all_tuning_frame)
plotable = plotable[plotable['Orien']!= 'False']
plotable['Orien'] = plotable['Orien'].str.slice(start=5)
plotable['Orien'] = plotable['Orien'].astype('f8')

plt.clf()
plt.cla()
fig, ax = plt.subplots(figsize = (5,4),dpi = 180)
sns.histplot(data = plotable,x = 'Orien',ax = ax,bins = np.linspace(0,180,9))
# sns.countplot(data = plotable,x = 'Orien',ax = ax,hue = 'Loc',legend = False)
ax.set_xlabel('Best Orien')

#%% color
plotable = copy.deepcopy(all_tuning_frame)
# plotable = plotable[plotable['Orien']!= 'False']
# plotable[plotable['Color']=='False'] = 'White'
plt.clf()
plt.cla()
fig, ax = plt.subplots(figsize = (5,4),dpi = 180)
sns.histplot(data = plotable,x = 'Color',ax = ax)
# sns.countplot(plotable, x="Color",ax =ax,hue = 'Loc',legend = False)
ax.set_xlabel('Best Color')