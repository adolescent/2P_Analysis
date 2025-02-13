'''
Calculate burstiness of each cell in spontaneous response, and get a statements.

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

all_path_dic = list(ot.Get_Subfolders(r'D:\#Fig_Data\_All_Spon_Data_V1'))

all_path_dic.pop(4)
all_path_dic.pop(6)
save_path = r'D:\_GoogleDrive_Files\#Figs\#250211_Revision1\Fig1'

#%% calculate burstiness of all cell
from scipy.signal import find_peaks,peak_widths
burstiness = pd.DataFrame(columns = ['Loc','Cell','Burstiness'])

for i,cloc in enumerate(all_path_dic):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    spon_start = c_spon.index[0]
    spon_end = c_spon.index[-1]
    c_spon = Z_refilter(ac,'1-001',spon_start,spon_end)
    for j in range(len(c_spon)):
        c_series = c_spon[j,:]
        peaks,_ = find_peaks(c_series, distance=3,height=-0.5) 
        c_raster = np.zeros(len(c_series))
        c_raster[peaks]=1
        cc_bur = Burstiness_Index(c_raster)
        burstiness.loc[len(burstiness)] = [cloc_name,j+1,cc_bur]
ot.Save_Variable(save_path,'Burstiness',burstiness)
#%% Plot burstiness

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4),dpi = 300,sharex= False)

sns.histplot(data = burstiness,x = 'Burstiness',bins = np.linspace(-0.55,0,40),ax = ax)
ax.axvline(x = burstiness['Burstiness'].mean(),linestyle='--',color = [0.7,0.7,0.7])
fig.savefig(ot.join(save_path,'Fig1Q_Burstiness.png'),bbox_inches='tight')
