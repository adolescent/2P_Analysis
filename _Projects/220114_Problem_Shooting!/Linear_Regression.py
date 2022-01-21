# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 10:22:08 2022

@author: ZR
"""
import OS_Tools_Kit as ot
import numpy as np
from Stimulus_Cell_Processor.Cell_Info_Cross_Corr import Correlation_Core
import matplotlib.pyplot as plt
from Series_Analyzer.Single_Component_Visualize import Single_Comp_Visualize,Single_Mask_Visualize
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
import pandas as pd
from tqdm import tqdm
import scipy.stats as stats
import seaborn as sns
from Stimulus_Cell_Processor.Get_Tuning import Get_Tuned_Cells
import List_Operation_Kit as lt
import random
from Analyzer.Statistic_Tools import T_Test_Pair
from sklearn.linear_model import LinearRegression
from Cell_Tools.Cell_Dist_Map import Cell_Dist_Map
from tqdm import tqdm
from Series_Analyzer.Cell_Frame_PCA import Do_PCA,PCA_Regression
#%% Read in necessary data here.
day_folder = r'G:\Test_Data\2P\210831_L76_2P'
Cell_dist_frame = Cell_Dist_Map(day_folder)
ot.Save_Variable(day_folder, 'Cell_Distance', Cell_dist_frame,'.dist')
Run01_Frame = Pre_Processor(day_folder,start_time = 7000)
acn = list(Run01_Frame.index)
all_tunings = ot.Load_Variable(day_folder,'All_Tuning_Property.tuning')
LE_cells = Get_Tuned_Cells(day_folder, 'LE',thres = 0.01)
LE_cells = list(set(LE_cells)&set(acn))
LE_cells.sort()
#%% Get PC 1 regressed result.
comp,info,weights = Do_PCA(Run01_Frame)
regressed_PCA = PCA_Regression(comp, info, weights,
                               ignore_PC = [1],var_ratio = 0.75)


#%% Calculate data for regression.
Corr_Matrix = pd.DataFrame(columns = ['Cell_A','Cell_B','Pearsonr','Dist','OD_Tuning_diff','OD_Tuning_multiplex','All_Tuning_Similarity'])
counter = 0
for i in tqdm(range(len(acn))):
    cell_A = acn[i]
    for j in range(i+1,len(acn)):
        cell_B = acn[j]
        c_r,_ = stats.pearsonr(Run01_Frame.loc[cell_A],Run01_Frame.loc[cell_B])
        c_dist = Cell_dist_frame.loc[cell_A,cell_B]
        c_tuning_diff = abs(all_tunings[cell_A]['LE']['Cohen_D']-all_tunings[cell_B]['LE']['Cohen_D'])
        c_tuning_multiplex = all_tunings[cell_A]['LE']['Cohen_D']*all_tunings[cell_B]['LE']['Cohen_D']
        all_tuning_A = np.array([all_tunings[cell_A]['LE']['Cohen_D'],all_tunings[cell_A]['Orien0']['Cohen_D'],all_tunings[cell_A]['Orien45']['Cohen_D']])
        all_tuning_B = np.array([all_tunings[cell_B]['LE']['Cohen_D'],all_tunings[cell_B]['Orien0']['Cohen_D'],all_tunings[cell_B]['Orien45']['Cohen_D']])
        tuning_similar = np.dot(all_tuning_A,all_tuning_B)/(np.linalg.norm(all_tuning_A)*np.linalg.norm(all_tuning_B))
        Corr_Matrix.loc[counter] = [cell_A,cell_B,c_r,c_dist,c_tuning_diff,c_tuning_multiplex,tuning_similar]
        counter +=1
Corr_Matrix.loc[:,'1/Dist'] = 1/Corr_Matrix.loc[:,'Dist']
ot.Save_Variable(day_folder, 'Corr_Matrix', Corr_Matrix)
#%% Try linear regression model.
y = Corr_Matrix.loc[:,'Pearsonr']
x = Corr_Matrix.loc[:,['Dist','All_Tuning_Similarity']]
model = LinearRegression()
model.fit(x,y)
model.score(x,y)
plt.scatter(x.loc[:,'All_Tuning_Similarity'],y,s = 2)
#%% Group by cell tunings
LE_Corr_Matrix = pd.DataFrame()
a = dict(list(Corr_Matrix.groupby('Cell_A')))
for i,cc in enumerate(LE_cells):
    LE_Corr_Matrix = LE_Corr_Matrix.append(a[cc])

y = LE_Corr_Matrix.loc[:,'Pearsonr']
x = LE_Corr_Matrix.loc[:,['Dist','All_Tuning_Similarity']]
model = LinearRegression()
model.fit(x,y)
model.score(x,y)
plt.scatter(x.loc[:,'OD_Tuning_multiplex'],y,s = 2)


