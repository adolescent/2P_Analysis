# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 12:38:24 2021

@author: ZR
"""

from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
from Series_Analyzer.Cell_Frame_PCA import Do_PCA
from Series_Analyzer.Cell_Frame_PCA import Compoment_Visualize
import OS_Tools_Kit as ot


day_folder = r'F:\Test_Data\2P\210831_L76_2P'
save_folder = r'F:\Test_Data\2P\210831_L76_2P\_All_Results\PCA_Results'
all_cell_dic = ot.Load_Variable(day_folder,'L76_210831A_All_Cells.ac')
before_spon = Pre_Processor(day_folder)
before_PCs,before_PC_info,before_fitted_weights = Do_PCA(before_spon)
Compoment_Visualize(before_PCs, all_cell_dic, save_folder)



