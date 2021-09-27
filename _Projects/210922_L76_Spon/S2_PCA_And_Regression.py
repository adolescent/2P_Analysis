# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 15:33:48 2021

@author: ZR
"""

from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
import Series_Analyzer.Cell_Frame_PCA as My_PCA
import OS_Tools_Kit as ot

day_folder = r'F:\Test_Data\2P\210920_L76_2P'
output_folder = r'F:\Test_Data\2P\210920_L76_2P\_All_Results\PCA'
all_cell_dic = ot.Load_Variable(day_folder,'L76_210920A_All_Cells.ac')
Spon_Before_frame = Pre_Processor(day_folder)
before_pcs,before_pc_info,before_fitted_weights = My_PCA.Do_PCA(Spon_Before_frame)
My_PCA.Compoment_Visualize(before_pcs, all_cell_dic, output_folder)



