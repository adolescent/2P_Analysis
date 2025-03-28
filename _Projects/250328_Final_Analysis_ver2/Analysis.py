'''
This script will do simle examples for thesis graphs.

Just make it quick = =

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
savepath = r'D:\_GoogleDrive_Files\#卒论\Figs'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
sponrun = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
start = sponrun.index[0]
end = sponrun.index[-1]
ac.wp = savepath

#%%
'''
Fig 3-1, Getting color tuning map.
'''
# green_resp = ac.Color_t_graphs['Green-White'].loc['A_response']-ac.Color_t_graphs['Green-White'].loc['B_response']
# red_resp = ac.Color_t_graphs['Red-White'].loc['A_response']-ac.Color_t_graphs['Red-White'].loc['B_response']
# blue_resp = ac.Color_t_graphs['Blue-White'].loc['A_response']-ac.Color_t_graphs['Blue-White'].loc['B_response']

red_resp = ac.Color_t_graphs['Red-White'].loc['CohenD']
green_resp = ac.Color_t_graphs['Green-White'].loc['CohenD']
blue_resp = ac.Color_t_graphs['Blue-White'].loc['CohenD']

r = ac.Generate_Weighted_Cell(red_resp)
r = r.clip(-5,5)
r= (r*255/r.max()).astype('u1')
# r = (r.clip(-2.5,2.5)*255/2.5).astype('u1')
g = ac.Generate_Weighted_Cell(green_resp)
g = g.clip(-5,5)
g= (g*125/g.max()).astype('u1')
# g = (g.clip(-2.5,2.5)*255/2.5).astype('u1')
b = ac.Generate_Weighted_Cell(blue_resp)
b = b.clip(-5,5)
b= (b*255/b.max()).astype('u1')
# b = (b.clip(-2.5,2.5)*255/2.5).astype('u1')

graph = np.zeros(shape = (512,512,3),dtype='u1')
graph[:,:,0] = r
# graph[:,:,1] = g
graph[:,:,2] = b

plt.imshow(graph)
# sns.heatmap(graph,square=True,yticklabels=False,xticklabels=False,center = 0,cbar=False)


