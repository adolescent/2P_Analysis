# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 19:18:34 2022

@author: ZR
"""

from Caiman_API.Precess_Pipeline import Preprocess_Pipeline


wp = r'D:\ZR\_Temp_Data\220914_L85_2P'
pp = Preprocess_Pipeline(wp, [1,2,3,6,7,8],orien_run = 'Run007',color_run = 'Run008')
pp.Do_Preprocess()

from Series_Analyzer.Cell_Frame_PCA_Cai import One_Key_PCA
comp,info,weight = One_Key_PCA(wp, 'Run001',tag = 'Spon_Before',start_frame = 4000)
comp_a,info_a,weight_a = One_Key_PCA(wp, 'Run003',tag = 'Spon_After',start_frame = 0)
