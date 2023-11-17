'''
Whether we have new ensembles?
Try to determine which ensemble of these ones be like.
Try Different methods.

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
from Filters import Signal_Filter

work_path = r'D:\_Path_For_Figs\FigS2e_All_Orientation_Funcmap'
expt_folder = r'D:\_All_Spon_Datas_V1\L76_18M_220902'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
ac.wp = expt_folder
# Get stim label and stim response.
all_stim_frame,all_stim_label = ac.Combine_Frame_Labels(od = True,orien = True,color = True,isi = True)
spon_series = ot.Load_Variable(expt_folder,'Spon_Before.pkl')
reducer = ot.Load_Variable(expt_folder,'All_Stim_UMAP_3D_20comp.pkl')
#%%################# METHOD 1, ALL UMAP ON ENSEMBLES ##########################
stim_embeddings = reducer.transform(all_stim_frame)
spon_embeddings = reducer.transform(spon_series)

classifier,score = SVM_Classifier(embeddings=stim_embeddings,label = all_stim_label)
predicted_spon_label = SVC_Fit(classifier,data = spon_embeddings,thres_prob = 0)