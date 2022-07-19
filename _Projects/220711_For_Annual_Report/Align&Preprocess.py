# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:29:52 2022

@author: ZR
"""


from Caiman_API.One_Key_Caiman import One_Key_Caiman
import OS_Tools_Kit as ot
import Graph_Operation_Kit as gt
from Stim_Frame_Align import One_Key_Stim_Align
from Caiman_API.Condition_Response_Generator import All_Cell_Condition_Generator
from Caiman_API.Map_Generators_CAI import One_Key_T_Map
from Stimulus_Cell_Processor.Get_Cell_Tuning_Cai import Tuning_Calculator
import warnings
from Decorators import Timer
from Caiman_API.Precess_Pipeline import Preprocess_Pipeline
import numpy as np
import matplotlib.pyplot as plt


#%% process L91 data into usable format.
day_folder_91 = r'D:\ZR\_Temp_Data\220420_L91'
pp = Preprocess_Pipeline(day_folder_91, [1,2,3,6,7,8],orien_run = 'Run007',color_run = 'Run008',boulder = (20,20,20,40))
pp.Do_Preprocess()
#%% get tamplate cell fit model.
day_folder_76 = r'D:\ZR\_Temp_Data\220630_L76_2P'
work_path = r'D:\ZR\_Temp_Data\220630_L76_2P\_CAIMAN'
#tamplate_id = 8
all_cell_cond_resp = ot.Load_Variable(work_path,'Cell_Condition_Response.pkl')
all_cell_dic = ot.Load_Variable(work_path,'All_Series_Dic.pkl')
Cell_Tuning_Dic = ot.Load_Variable(work_path,'Cell_Tuning_Dic.pkl')
x_real = list(np.arange(0,360,22.5))
y_real = []
for i in range(1,17):
    c_cond = all_cell_cond_resp[8]['Run007'][i]
    y_real.append(c_cond[[4,5],:].mean())
parameters = Cell_Tuning_Dic[8]['Fit_Parameters']
x_sim_rad = np.arange(0,2*np.pi,0.01)
def Mises_Function(c_angle,best_angle,a0,b1,b2,c1,c2):
    y = a0+b1*np.exp(c1*np.cos(c_angle-best_angle))+b2*np.exp(c2*np.cos(c_angle-best_angle-np.pi))
    return y
y_sim = Mises_Function(x_sim_rad,parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5])
x_sim = x_sim_rad*180/np.pi
plt.plot(x_sim,y_sim)
plt.plot(x_real,y_real)
#%% calculate all fit difference.
acn = list(all_cell_cond_resp.keys())
tuning_diff = []
for i,cc in enumerate(acn):
    if Cell_Tuning_Dic[cc]['Orien_Preference'] != 'No_Tuning':
        c_best_tuning = float(Cell_Tuning_Dic[cc]['Orien_Preference'][5:])
        c_fitted_tuning = Cell_Tuning_Dic[cc]['Fitted_Orien']
        raw_tuning_diff = abs(c_best_tuning-c_fitted_tuning)%90
        tuning_diff.append(raw_tuning_diff)
        
#%% compare PC comp with stim maps.
from Series_Analyzer.Cell_Frame_PCA_Cai import One_Key_PCA
from scipy.stats import pearsonr

comp,info,weight = One_Key_PCA(day_folder_91,'Run001',start_frame = 3000,tag = 'Spon_Before')
comp_a,info_a,weight_A = One_Key_PCA(day_folder_91,'Run003',start_frame = 0,tag = 'Spon_After')
od_map = ot.Load_Variable(r'D:\ZR\_Temp_Data\220420_L91\_CAIMAN\Run006_T_Maps\All_Map_Response.pkl')['OD'].loc['t',:]
pc2 = comp['PC002']
r,p = pearsonr(od_map,pc2)
havo_map = ot.Load_Variable(r'D:\ZR\_Temp_Data\220420_L91\_CAIMAN\Run007_T_Maps\All_Map_Response.pkl')['HA-VO'].loc['t',:]
pc5 = comp['PC005']
r,p = pearsonr(havo_map,pc5)

