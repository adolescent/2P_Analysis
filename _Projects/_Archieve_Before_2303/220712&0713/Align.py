# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 19:23:38 2022

@author: ZR
"""

from Caiman_API.Precess_Pipeline import Preprocess_Pipeline
from Series_Analyzer.Cell_Frame_PCA_Cai import One_Key_PCA
import OS_Tools_Kit as ot


day_folder = r'G:\Test_Data\2P\220712_L76_2P'
pp = Preprocess_Pipeline(day_folder, [1,2,3,6,7],orien_run = 'Run002',color_run = 'Run007',boulder = (20,20,20,50))
pp.Do_Preprocess()

comp,weight,info = One_Key_PCA(day_folder, 'Run001',tag = 'Spon_Before',start_frame = 4000)
comp_a,weight_a,info_a= One_Key_PCA(day_folder, 'Run003',tag = 'Spon_After',start_frame = 0)
