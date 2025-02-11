'''
Answer the question of why we use low-pass filter, and what will happen if we don't filt.
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
from Kill_Cache import kill_all_cache
from sklearn.model_selection import cross_val_score
from sklearn import svm
# import umap
# import umap.plot
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
from Cell_Class.Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *



exp_path = r'D:\ZR\#FigDatas\_All_Spon_Data_V1\L76_SM_Run03bug_210721'
ac = ot.Load_Variable_v2(exp_path,'Cell_Class.pkl')
sponrun =  ot.Load_Variable_v2(exp_path,'Spon_Before.pkl')
start = sponrun.index[0]
end = sponrun.index[-1]
all_cell_dic = ac.all_cell_dic
fps = ac.fps

savepath = r'D:\ZR\_Data_Temp\_Article_Data\_Revised_Data'

#%% Generate new unfilted Z frames.
from Fix_Funcs import *
z_origin = Z_refilter(ac,start,end,'1-001',0.005,0.3)
z_no_lp = Z_refilter(ac,start,end,'1-001',0.005,False)
vmax = 4
vmin = -2

fig,ax = plt.subplots(ncols=1,nrows=2,figsize = (8,6),dpi=300,sharex=True)
sns.heatmap(z_origin[:,3000:3650],ax = ax[0],cbar=False,xticklabels=False,yticklabels=False,center = 0,vmax = vmax,vmin = vmin)
sns.heatmap(z_no_lp[:,3000:3650],ax = ax[1],cbar=False,xticklabels=False,yticklabels=False,center = 0,vmax = vmax,vmin = vmin)

fig.tight_layout()





#%% Freq analysis


# freq,power,_,_,total = FFT_Spectrum(z_train,1.301,0.01,False)
# freq_matrix = np.zeros(shape = (len(freq),len(all_cell_dic)),dtype = 'f8')
# for i in tqdm(range(len(z_frame))):
#     c_train = z_frame[i,:]
#     _,c_power,_,_,_ = FFT_Spectrum(c_train,1.301,0.01,False)
#     freq_matrix[:,i] = c_power

# sns.heatmap(freq_matrix.T,center=0,vmax = 0.1)