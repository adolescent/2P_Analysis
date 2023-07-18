'''
This script will sum all tunings of current spon frame, 

'''
#%%
import OS_Tools_Kit as ot
import numpy as np
import pandas as pd
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
import Graph_Operation_Kit as gt
import cv2
import umap
import umap.plot
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from Kill_Cache import kill_all_cache
from sklearn.model_selection import cross_val_score
from sklearn import svm
from Cell_Tools.Cell_Visualization import Cell_Weight_Visualization
import random
import seaborn as sns
from My_Wheels.Cell_Class.Stim_Calculators import Stim_Cells
from My_Wheels.Cell_Class.Format_Cell import Cell
from Cell_Class.Advanced_Tools import *
from Cell_Class.Plot_Tools import *
from Stim_Frame_Align import One_Key_Stim_Align
from scipy.stats import pearsonr
import scipy.stats as stats

all_path = ot.Get_Sub_Folders(r'D:\ZR\_Data_Temp\_All_V1_Before_Cell_Classes')
all_path = np.delete(all_path,[4,7]) # delete 2 point with not so good OD.
stats_path = r'D:\ZR\_Data_Temp\_Stats'
cp = all_path[0]
#%%
# get cell tunings.
ac = ot.Load_Variable(cp,'Cell_Class.pkl')
spon_frame = ot.Load_Variable(cp,'Spon_Before.pkl')
# spon_frame = spon_frame*(spon_frame>2)
# get current point cell tunings.
c_LE_cells = []
c_RE_cells = []
c_orien0_cells = []
c_orien45_cells = []
c_orien90_cells = []
c_orien135_cells = []
for j,cc in enumerate(ac.acn):
    c_response = ac.all_cell_tunings[cc]
    # get OD
    if c_response['Best_Eye'] == 'LE':
        c_LE_cells.append(cc)
    elif c_response['Best_Eye'] == 'RE':
        c_RE_cells.append(cc)
    # get HVAO 4 oriens.
    orien_deter = c_response[['Orien0-0','Orien45-0','Orien90-0','Orien135-0']]
    max_orien = orien_deter[orien_deter == orien_deter.max()].index[0]
    if ac.all_cell_tunings_p_value[cc][max_orien]<0.05:
        if max_orien == 'Orien0-0':
            c_orien0_cells.append(cc)
        elif max_orien == 'Orien45-0':
            c_orien45_cells.append(cc)
        elif max_orien == 'Orien90-0':
            c_orien90_cells.append(cc)
        elif max_orien == 'Orien135-0':
            c_orien135_cells.append(cc)
# get network response of all cells. 
LE_train = spon_frame.loc[:,c_LE_cells].mean(1)
RE_train = spon_frame.loc[:,c_RE_cells].mean(1)
orien0_train = spon_frame.loc[:,c_orien0_cells].mean(1)
orien45_train = spon_frame.loc[:,c_orien45_cells].mean(1)
orien90_train = spon_frame.loc[:,c_orien90_cells].mean(1)
orien135_train = spon_frame.loc[:,c_orien135_cells].mean(1)
all_train = spon_frame.loc[:,:].mean(1)
#%% plot
plt.switch_backend('webAgg')
# plt.plot(orien0_train)
# plt.plot(orien90_train)
# plt.scatter(orien0_train,orien90_train,s = 3)
plt.scatter(LE_train,RE_train,s = 3)
plt.plot([0,1,2,3,4],[0,1,2,3,4],linestyle = 'dashed',color = 'r')
plt.show()
#%% granger causality test
from statsmodels.tsa.stattools import grangercausalitytests
maxlag = 20  # maximum lag to test
results_xy = grangercausalitytests(np.array([LE_train,RE_train]).T, [50])
# results_yx = grangercausalitytests(np.array([RE_train,LE_train]).T, )
#%% slide window corr
lag = 500
# Compute sliding window correlation
x = LE_train
y = RE_train
corr_list = []
for i in range(lag):
    c_x_train = np.array(x[:-lag])
    c_y_train = np.array(y[i:-lag+i])
    c_corr,c_p = stats.pearsonr(c_x_train,c_y_train)
    corr_list.append(c_corr)
plt.switch_backend('webAgg')
plt.plot(corr_list)
plt.show()
#%% cross power spectrum
import numpy as np
from scipy.signal import csd
frequencies, cross_power_spectrum = csd(LE_train,RE_train,fs = 1.301)
plt.switch_backend('webAgg')
plt.semilogy(frequencies, np.abs(cross_power_spectrum))
# plt.plot(frequencies,np.abs(cross_power_spectrum))
plt.xlabel('frequency [Hz]')
plt.ylabel('CSD')
plt.show()