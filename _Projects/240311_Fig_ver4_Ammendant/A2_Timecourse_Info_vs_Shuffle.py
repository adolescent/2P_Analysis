'''
This script will show the wait time with random selection.
'''


#%% Initialization
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


wp = r'D:\_Path_For_Figs\240312_Figs_v4_A1\A1_Waittime_Distribution'
all_waittime = ot.Load_Variable(wp,'All_Network_Waittime.pkl')
all_repeat_series = ot.Load_Variable(wp,'All_Repeat_Series.pkl')
all_loc_names = list(all_repeat_series.keys())



