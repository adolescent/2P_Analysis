# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 14:24:21 2022

@author: ZR

This script is used to generate Dist&Tuning regression on different time windows.

"""


from Series_Analyzer.Preprocessor_Cai import Pre_Processor_Cai
import OS_Tools_Kit as ot
from Series_Analyzer.Pairwise_Correlation import Series_Cut_Pair_Corr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm




#%% Initailization
wp = r'D:\ZR\_Temp_Data\220711_temp'

pcinfo76 = ot.Load_Variable(wp,'pc76_info.pkl')
pcwin76 = ot.Load_Variable(wp,'pc76win.pkl')
pcwin76_used = pcwin76.loc[:,60:]
tempwin = pcwin76_used.loc[:,74]

#%% We used OD diff, dist and other tuning to make correlation here.






