'''
This is a data generator for all countable data.

Remember to get a definition of when to start counting spon.
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

#%% L76-15A 220812
wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220812_L76_2P'
ac = Stim_Cells(day_folder = wp,od =6,orien =2,color = 7)
ac.Calculate_All()
ac.Save_Class()
#%% L76-18M 220902
wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220902_L76_2P'
ac = Stim_Cells(day_folder = wp,od =6,orien =7,color = 8)
ac.Calculate_All()
ac.Save_Class()
#%% L85-17B-220727
wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220727_L85_2P'
ac = Stim_Cells(day_folder = wp,od =6,orien =7,color = 8)
ac.Calculate_All()
ac.Save_Class()
#%% L85-19B-220713
wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220713_L85_2P'
ac = Stim_Cells(day_folder = wp,od =6,orien =2,color = 7)
ac.Calculate_All()
ac.Save_Class()
#%% L91-11A-220609 OD problem.
wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220609_L91_2P'
One_Key_Stim_Align(r'D:\ZR\_Data_Temp\Raw_2P_Data\220609_L91_2P\220609_L91_stimuli')
ac = Stim_Cells(day_folder = wp,od =6,orien =7,color = 8)
ac.Calculate_All()
ac.Plot_T_Graphs()
ac.Save_Class()
#%% L91-8A-220504 Boulder problem.
wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\220504_L91'
ac = Stim_Cells(day_folder = wp,od =6,orien =2,color = 7)
ac.Calculate_All()
# ac.Plot_T_Graphs()
ac.Save_Class()
#%% L76-SM-210721
wp = r'D:\ZR\_Data_Temp\Raw_2P_Data\210721_L76_2P'
ac = Stim_Cells(day_folder = wp,od =6,orien =2,color = 7)
ac.Calculate_All()
ac.Plot_T_Graphs()
ac.Save_Class()
#%% And All 3 V2 points.
#%% 220211 L76 Loc 6B
wp = r'E:\2P_Raws\220211_L76_2P'
ac = Stim_Cells(day_folder = wp,od =False,orien =2,color = 5,od_type = False)
ac.Calculate_All()
ac.Plot_T_Graphs()
ac.Save_Class()
#%% 220825 L85 Loc 6B
wp = r'E:\2P_Raws\220825_L85_2P'
ac = Stim_Cells(day_folder = wp,od =6,orien =2,color = 7)
ac.Calculate_All()
ac.Plot_T_Graphs()
ac.Save_Class()
#%% 230808 L76 Loc 7A
wp = r'E:\2P_Raws\230808_L76_2P'
ac = Stim_Cells(day_folder = wp,od =6,orien =2,color = 7)
ac.Calculate_All()
ac.Plot_T_Graphs()
ac.Save_Class()


