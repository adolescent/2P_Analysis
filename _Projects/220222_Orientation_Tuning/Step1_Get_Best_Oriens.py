# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:25:15 2022

@author: ZR
"""

import OS_Tools_Kit as ot
import cv2
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
from Series_Analyzer.Cell_Frame_PCA import Do_PCA,PCA_Regression
import matplotlib.pyplot as plt
from Series_Analyzer.Single_Component_Visualize import Single_Mask_Visualize
from Stimulus_Cell_Processor.Get_Tuning import Get_Tuned_Cells
import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% read in 
day_folder = r'G:\Test_Data\2P\210831_L76_2P'
all_cell_dic = ot.Load_Variable(day_folder,'L76_210831A_All_Cells.ac')
Run01_Frame = Pre_Processor(day_folder,start_time = 7000)
acn = list(Run01_Frame.index)
G16_Run = 'Run002'
#%% Get average response.
tc = all_cell_dic[acn[23]]
used_frame = [4,5]
c_G16_Data = tc[G16_Run]['CR_Train']
c_angle = np.array(range(1,17))*22.5*np.pi/180
c_response = np.zeros(16)
tc_response = pd.DataFrame(columns = ['angle','response'])
counter = 0

for i in range(1,17):
    c_data = c_G16_Data[i][:,used_frame]
    for j in range(c_data.shape[0]):
        tc_response.loc[counter] = [(i-1)*22.5*np.pi/180,c_data[j,:].mean()]
        counter +=1
    

# fit response data with given function.
def Mises_Function(c_angle,best_angle,a0,b1,b2,c1,c2):
    # Please input angle in radius.
    y = a0+b1*np.exp(c1*np.cos(c_angle-best_angle))+b2*np.exp(c2*np.cos(c_angle-best_angle-np.pi))
    return y

from scipy.optimize import curve_fit
parameters, covariance = curve_fit(Mises_Function, tc_response.loc[:,'angle'],tc_response.loc[:,'response'])
from sklearn.metrics import r2_score
y_fit = tc_response.loc[:,'response']
y_pred = Mises_Function(tc_response.loc[:,'angle'],parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5])
r2_score(y_fit, y_pred)



#%%
a = Mises_Function(c_angle,parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5])
c_angle = np.arange(0,2*np.pi,0.01)


