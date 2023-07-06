'''
This script will find the determine point of all spon reaction.
Maybe it's okay to use just last 1h, but try to find whether there is another standard.
'''


#%%
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

all_path = ot.Get_Sub_Folders(r'D:\ZR\_Data_Temp\_All_Cell_Classes')
#%% get specific Run01 Frames.
cp = all_path[3]
ac = ot.Load_Variable(cp,'Cell_Class.pkl')
spon_frame = ac.Z_Frames['1-001']
spon_avr = spon_frame.mean(1)
reaction_cells = (spon_frame>2).sum(1)
#%% plot parts
plt.switch_backend('webAgg')
plt.plot(spon_avr)
# plt.plot(reaction_cells)
plt.show()
#%% Save 
used_spon_frame = spon_frame.loc[0:9100,:]
ot.Save_Variable(cp,'Spon_Before',used_spon_frame)

