'''
This part will check the co-activation of each network, and try to compare it with random?
'''

#%% Load in for example loc.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import cv2
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")


from Cell_Class.Stim_Calculators import Stim_Cells
from Cell_Class.Format_Cell import Cell
import OS_Tools_Kit as ot
from Cell_Class.Advanced_Tools import *
from Classifier_Analyzer import *

expt_folder = r'D:\_All_Spon_Data_V1\L76_18M_220902'
# save_path = r'D:\_GoogleDrive_Files\#Figs\240627_Figs_FF1\Fig1'
ac = ot.Load_Variable_v2(expt_folder,'Cell_Class.pkl')
c_spon = np.array(ot.Load_Variable(expt_folder,'Spon_Before.pkl'))
all_path_dic = list(ot.Get_Subfolders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)
all_path_dic_v2 = list(ot.Get_Subfolders(r'D:\_All_Spon_Data_V2'))
spon_dff = ac.Get_dFF_Frames('1-001',0.1,8500,13852)
#%%
for i,cloc in enumerate(all_path_dic):
    c_repeat = ot.Load_Variable(cloc,'All_Spon_Repeats_PCA10.pkl')


