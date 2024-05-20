# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:43:23 2022

@author: ZR
"""

from Series_Analyzer.Preprocessor_Cai import Pre_Processor_Cai
import OS_Tools_Kit as ot
from Series_Analyzer.Pairwise_Correlation import Series_Cut_Pair_Corr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr,spearmanr
import statsmodels.api as sm
import pandas as pd
from tqdm import tqdm
from Series_Analyzer.Series_Cutter import Series_Window_Slide
import cv2

wp = r'D:\ZR\_Temp_Data\220711_temp'
#%%
Run01_frame = ot.Load_Variable(wp,'Series91_Run01_0.pkl')
Run03_frame = ot.Load_Variable(wp,'Series91_Run03_0.pkl')
cell_tuning_dic = ot.Load_Variable(wp,r'Cell_Tuning_Dic91.pkl')
from Series_Analyzer.Response_Info import Get_Frame_Response
Run01_frame_response,cell_num_dic,actune = Get_Frame_Response(Run01_frame, cell_tuning_dic)

# thres Run01 and Run03 to get thresed frame.
# then get firing rate of each network.
from Series_Analyzer.Series_Cutter import Series_Window_Slide

