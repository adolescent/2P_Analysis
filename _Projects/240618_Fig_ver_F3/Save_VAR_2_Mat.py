'''
This will save vars into mat format for cross validation.
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

wp = r'D:\_All_Spon_Data_V1\L76_18M_220902'
ac = ot.Load_Variable(wp,'Cell_Class.pkl')
spon_series = ot.Load_Variable(wp,'Spon_Before.pkl')
# savepath = r'D:\_Path_For_Figs\240614_Figs_ver_F2\Fig2'
c_tunings = ac.all_cell_tunings

#%% save into mat.
ot.Save_Variable(r'D:\_GoogleDrive_Files\#Figs\Comments240621_Figs_ver_F4','Spon',spon_series)
ot.Save_Variable(r'D:\_GoogleDrive_Files\#Figs\Comments240621_Figs_ver_F4','Tuning',c_tunings)