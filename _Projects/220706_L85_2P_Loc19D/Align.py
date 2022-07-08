# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 20:09:08 2022

@author: ZR
"""


from My_Wheels.Caiman_API.Precess_Pipeline import Preprocess_Pipeline

day_folder = r'D:\ZR\_Temp_Data\220706_L85_LM'
pp = Preprocess_Pipeline(day_folder,[1,2,3,6,7,8],orien_run = 'Run007',color_run = 'Run008')
pp.Do_Preprocess()

#%% PCA here.
from My_Wheels.Series_Analyzer.Cell_Frame_PCA_Cai import One_Key_PCA
comp_b,info_b,weight_b = One_Key_PCA(day_folder,'Run001',start_frame = 4000,tag = 'Spon_Before')
comp_a,info_a,weight_a = One_Key_PCA(day_folder,'Run003',start_frame = 0,tag = 'Spon_After')
#%% wash cell again.
import OS_Tools_Kit as ot
import numpy as np
import Graph_Operation_Kit as gt

work_path = r'D:\ZR\_Temp_Data\220706_L85_LM\_CAIMAN'
all_cell_dic = ot.Load_Variable(r'D:\ZR\_Temp_Data\220706_L85_LM\_CAIMAN','All_Series_Dic.pkl')
x_range = [10,462]
y_range = [20,492]
acn = list(all_cell_dic.keys())
washed_cell_dic = {}
cell_mask = np.zeros(shape = (512,512),dtype = 'f8')
for i,cc in enumerate(acn):
    cc_dic = all_cell_dic[cc]
    cc_loc = cc_dic['Cell_Loc']
    if cc_loc[0]>y_range[0] and cc_loc[0]<y_range[1]:
        if cc_loc[1]>x_range[0] and cc_loc[1]<x_range[1]:
            washed_cell_dic[cc] = cc_dic
            cell_mask += cc_dic['Cell_Mask']
# save washed cells.
ot.Save_Variable(work_path, 'All_Series_Dic_Washed', washed_cell_dic)
cellmap_new = gt.Clip_And_Normalize(cell_mask,clip_std = 5)
gt.Show_Graph(cellmap_new, 'Cell_Location_New', work_path)
    
