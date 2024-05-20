'''
This script will calculate all repeat series, return an file of stim ids.

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
from Classifier_Analyzer import *
from Cell_Class.Timecourse_Analyzer import *



all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)

import warnings
warnings.filterwarnings("ignore")

#%% calculate all stim graphs.
all_repeat_frame = {}
for i,cloc in tqdm(enumerate(all_path_dic)):
    cloc_name = cloc.split('\\')[-1]
    ac = ot.Load_Variable_v2(cloc,'Cell_Class.pkl')
    c_spon = ot.Load_Variable(cloc,'Spon_Before.pkl')
    _,_,spon_models = Z_PCA(Z_frame=c_spon,sample='Frame',pcnum=10)
    c_repeat_frame = pd.DataFrame(0,columns = ['OD','Orien','Color'],index = range(len(c_spon)))

    # Orien
    analyzer_orien = Classify_Analyzer(ac = ac,umap_model=spon_models,spon_frame=c_spon,od = 0, color = 0,orien = 1)
    analyzer_orien.Train_SVM_Classifier()
    orien_labels = analyzer_orien.spon_label
    c_repeat_frame.loc[:,'Orien'] = orien_labels

    #OD
    analyzer_od = Classify_Analyzer(ac = ac,umap_model=spon_models,spon_frame=c_spon,od = 1, color = 0,orien = 0)
    analyzer_od.Train_SVM_Classifier()
    od_labels = analyzer_od.spon_label
    c_repeat_frame.loc[:,'OD'] = od_labels

    #Color
    analyzer_color = Classify_Analyzer(ac = ac,umap_model=spon_models,spon_frame=c_spon,od = 0, color = 1,orien = 0)
    analyzer_color.Train_SVM_Classifier()
    color_labels = analyzer_color.spon_label
    c_repeat_frame.loc[:,'Color'] = color_labels
    ot.Save_Variable(cloc,'All_Spon_Repeats_PCA10',c_repeat_frame)
    all_repeat_frame[cloc_name] = c_repeat_frame
ot.Save_Variable(r'D:\_Path_For_Figs\240401_Figs_v6','All_Spon_Repeats_All',all_repeat_frame)