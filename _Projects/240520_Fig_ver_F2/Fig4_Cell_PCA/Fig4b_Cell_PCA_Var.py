

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

work_path = r'D:\_Path_For_Figs\240520_Figs_ver_F1\Fig4_Cell_In_Spon'
all_path_dic = list(ot.Get_Subfolders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)

#%%
'''
Redo Fig 4B. just get error bar for all locs explatined var.
'''
used_pc_num = 20
all_explained_var = np.zeros(shape=(used_pc_num,len(all_path_dic)))

for i,cloc in tqdm(enumerate(all_path_dic)):
    c_spon = np.array(ot.Load_Variable_v2(cloc,'Spon_Before.pkl'))
    c_ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_pc_comps,c_pc_coords,c_pc_model = Z_PCA(c_spon,sample='Cell',pcnum=used_pc_num)
    print(f'Total {used_pc_num} explained VAR {c_pc_model.explained_variance_ratio_.sum()*100:.1f}%')
    all_explained_var[:,i] = c_pc_model.explained_variance_ratio_

#%% Plot parts
plotable = pd.DataFrame(all_explained_var.T).melt(var_name='PC',value_name='Explained VAR Ratio')
plotable['Explained VAR Ratio'] = plotable['Explained VAR Ratio']*100
plotable['PC'] = plotable['PC']+1

plt.clf()
plt.cla()
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (6,4),dpi = 180)
sns.barplot(data = plotable,y = 'Explained VAR Ratio',x = 'PC',ax = ax,capsize=0.2)
ax.set_xlabel('PC',size = 12)
ax.set_ylabel('Explained Ratio(%)',size = 12)
# ax.set_title('Each PC explained Variance',size = 14)
ax.set_ylim(0,7)

