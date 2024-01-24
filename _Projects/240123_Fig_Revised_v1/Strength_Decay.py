'''
This part will give a ensemble size or activation strength drop of all location datas.

This will be used on determine stim on locations.
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
import umap
import umap.plot
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
from Cell_Class.UMAP_Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *


work_path = r'D:\_Path_For_Figs\240123_Graph_Revised_v1\Fig1_Revised'
expt_folder = r'D:\_All_Spon_Data_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
ac.wp = expt_folder
global_avr = cv2.imread(f'{expt_folder}\Global_Average_cai.tif',0) # read as 8 bit gray scale map.
c_spon = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
ac.Regenerate_Cell_Graph()

import warnings
warnings.filterwarnings("ignore")