'''
DECREPTED, NOT IMPORTANT.
CAN USE IT IF WE NEEE.

This script will show the recover of all orientation (Not only the give 8 orientations)
We will find that all orientation have been repeated.

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

work_path = r'D:\_Path_For_Figs\240520_Figs_ver_F1\Fig3_PCA_SVM'
all_path_dic = list(ot.Get_Subfolders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
# some times we need to ignore warnings.
import warnings
warnings.filterwarnings("ignore")

# here we load directly. We can calculate it easily.
var_path = r'D:\_Path_For_Figs\240520_Figs_ver_F1\Fig3_PCA_SVM\VARs_From_v4'
all_orien_maps = ot.Load_Variable(var_path ,'VAR1_All_Orien_Response.pkl')
all_cell_oriens = ot.Load_Variable(var_path ,'VAR2_All_Cell_Best_Oriens.pkl')


#%% 
'''

'''