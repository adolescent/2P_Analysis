# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 20:27:25 2022

@author: ZR
"""

from Caiman_API.Precess_Pipeline import Preprocess_Pipeline

day_folder = r'D:\ZR\_Temp_Data\220727_L85_2P'

pp = Preprocess_Pipeline(day_folder,[1,2,3,6,7,8],orien_run='Run007',color_run='Run008')
pp.Do_Preprocess()


from Series_Analyzer.Cell_Frame_PCA_Cai import One_Key_PCA
One_Key_PCA(day_folder, 'Run001',tag = 'Spon_Before',start_frame = 4000)
One_Key_PCA(day_folder, 'Run003',tag = 'Spon_After',start_frame = 0)
