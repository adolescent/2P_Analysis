
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

# from Kill_Cache import kill_all_cache
from sklearn.model_selection import cross_val_score
from sklearn import svm
from scipy.stats import pearsonr
import scipy.stats as stats
from Cell_Class.Plot_Tools import Plot_3D_With_Labels
import copy
from Cell_Class.Advanced_Tools import *
import random

wp = r'D:\_GoogleDrive_Files\#Figs\#250211_Revision1\Fig3'

#%%
hue_freq = ot.Load_Variable_v2(wp,'Hue_Repeat_Freq.pkl')
od_freq = ot.Load_Variable_v2(wp,'OD_Repeat_Freq.pkl')
orien_freq = ot.Load_Variable_v2(wp,'Orien_Repeat_Freq.pkl')

