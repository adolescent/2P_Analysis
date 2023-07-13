'''
This script will try to compare spon before stim and after stim, to see whether we have comparable differences.
'''


#%% imports 
import OS_Tools_Kit as ot
import numpy as np
import pandas as pd
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
import Graph_Operation_Kit as gt
import cv2
import umap
import umap.plot
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from Kill_Cache import kill_all_cache
from sklearn.model_selection import cross_val_score
from sklearn import svm
from Cell_Tools.Cell_Visualization import Cell_Weight_Visualization
import random
import seaborn as sns
from My_Wheels.Cell_Class.Stim_Calculators import Stim_Cells
from My_Wheels.Cell_Class.Format_Cell import Cell
from Cell_Class.Advanced_Tools import *
from Cell_Class.Plot_Tools import *
from Stim_Frame_Align import One_Key_Stim_Align
from scipy.stats import pearsonr
import scipy.stats as stats


wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220211_L76_2P'
ac = Stim_Cells(day_folder = wp,od = False,od_type = False,orien =2,color = 5,filter_para = (0.005,0.3))
ac.Calculate_All()
ac.Save_Class()
# ac.Plot_T_Graphs()
#%% Plot spon
spon_frames = ac.Z_Frames['1-001']
plt.switch_backend('webAgg')
plt.plot(spon_frames.mean(1))
plt.show()
