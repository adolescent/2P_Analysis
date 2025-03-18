'''
Shows tuning curve of a given cell, getting response 

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
from Cell_Class.Timecourse_Analyzer import *
from Review_Fix_Funcs import *
from Filters import Signal_Filter_v2
import warnings


warnings.filterwarnings("ignore")

expt_folder = r'D:\#Fig_Data\_All_Spon_Data_V1\L76_18M_220902'
savepath = r'D:\_GoogleDrive_Files\#卒论\Figs\Fig3-1'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
sponrun = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
start = sponrun.index[0]
end = sponrun.index[-1]
ac.wp = savepath
#%%
a = ac.Annotate_Cell([40,150,396])
fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(4,4),dpi=300)
ax.imshow(a)
ax.set_yticks([])
ax.set_xticks([])

#%% ####################### PLOT FUNC MAPS
## Orien map
ac_locs = ac.Get_Cell_Loc()
all_best_oriens = ot.Load_Variable(r'D:\_GoogleDrive_Files\#Figs\#240802_Figs_Ver_CR&Elife\#Figs\Fig4\All_Cell_Best_Oriens.pkl')
ac_tunes = all_best_oriens['L76_18M_220902']
#%% 
import matplotlib.patches as patches
from colorsys import hsv_to_rgb

fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=300)
ax.set_xlim(0, 512)
ax.set_ylim(0, 512)
ax.set_aspect('equal')
ax.axis('off')  # Hide axes
# ax.set_facecolor('black')

for i in range(1,len(ac)+1):
    yi = ac_locs[i]['Y']
    xi = ac_locs[i]['X']
    c_angle = ac_tunes.loc[i]['Best_Angle']
    if c_angle != -1:
        rgb_color = hsv_to_rgb(c_angle/180, 1, 1)
        # Create circle with radius 1.5 (3 pixels diameter)
        circle = patches.Circle(
            (yi, xi),
            radius=3.5,  # 3 pixel diameter
            facecolor=rgb_color,
            edgecolor='b',
            linewidth=0
        )
        ax.add_patch(circle)
# Remove padding around the plot
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()

#%%  OD
od_resp = ac.OD_t_graphs['OD'].loc['CohenD']
od_graph = ac.Generate_Weighted_Cell(od_resp)
od_graph = od_graph.clip(-2.5,2.5)
sns.heatmap(od_graph,square=True,yticklabels=False,xticklabels=False,center = 0,cbar=False)

#%%
green_resp = ac.Color_t_graphs['Green-White'].loc['A_response']-ac.Color_t_graphs['Green-White'].loc['B_response']

red_resp = ac.Color_t_graphs['Red-White'].loc['A_response']-ac.Color_t_graphs['Red-White'].loc['B_response']