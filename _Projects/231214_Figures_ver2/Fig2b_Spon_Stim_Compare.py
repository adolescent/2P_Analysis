'''
This graph compare same graph taken from spon and stimulus stim graph.
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

work_path = r'D:\_Path_For_Figs\_2312_ver2\Fig2'
expt_folder = r'D:\_All_Spon_Datas_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
ac.wp = expt_folder
spon_series = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
# load reducer, if not exist, generate a new one.
reducer = ot.Load_Variable_v2(expt_folder,'All_Stim_UMAP_3D_20comp.pkl')

#%%################## STEP1, GET COMPARE GRAPHS #########################
analyzer = UMAP_Analyzer(ac = ac,umap_model=reducer,spon_frame=spon_series,od = True,orien = True,color = True,isi = True)
analyzer.Train_SVM_Classifier()
analyzer.Get_Stim_Spon_Compare()
compare_graphs = analyzer.compare_recover
