
'''
This graph will do stats to all data points, getting stimulus recovered map similarity with stim pattern.
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

work_path = r'D:\_Path_For_Figs\_2312_ver2\Fig2'
all_path_dic = list(ot.Get_Sub_Folders(r'D:\_All_Spon_Data_V1'))
all_path_dic.pop(4)
all_path_dic.pop(6)

#%%#################### STEP1, GET ALL COMPARE DIC #######################

for i,c_loc in tqdm(enumerate(all_path_dic)):
    c_loc_last = c_loc.split('\\')[-1]
    c_spon_frame = ot.Load_Variable(c_loc,'Spon_Before.pkl')
    c_ac = ot.Load_Variable_v2(c_loc,'Cell_Class.pkl')
    c_reducer = ot.Load_Variable(c_loc,'All_Stim_UMAP_3D_20comp.pkl')
    c_analyzer = Classify_Analyzer(ac = c_ac,umap_model=c_reducer,spon_frame=c_spon_frame)
    c_analyzer.Similarity_Compare_Average()
    c_recover_similar = c_analyzer.Avr_Similarity
    c_recover_similar['Location'] = c_loc_last
    if i == 0:
        all_recover_similarity = copy.deepcopy(c_recover_similar)
    else:
        all_recover_similarity = pd.concat([all_recover_similarity,c_recover_similar],ignore_index=True)

#%%############### STEP2, GROUP BY TUNING AND SHUFFLE. ##########################
plt.clf()
plt.cla()
# set graph
fig,ax = plt.subplots(nrows=1, ncols=1,figsize = (2.5,5),dpi = 180)
ax.axhline(y = 0,color='gray', linestyle='--')
sns.boxplot(data = all_recover_similarity,x = 'Data',y = 'PearsonR',hue = 'MapType',ax = ax,showfliers = False)
ax.set_title('Network Repeat Similarity')
ax.set_xlabel('')
ax.legend(title = 'Network',fontsize = 8)
ax.set_xticklabels(['Real Data','Random Select'],size = 7)
ax.set_ylabel('Pearson R')
plt.show()