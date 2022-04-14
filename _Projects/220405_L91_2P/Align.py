# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 20:11:52 2022

@author: ZR
"""
import matplotlib.pyplot as plt
from Standard_Aligner import Standard_Aligner
day_folder = r'G:\Test_Data\2P\220405_L91'
Sa = Standard_Aligner(day_folder, [1,2,3,4,5,6,7,8],final_base = '1-003')
Sa.One_Key_Aligner_No_Affine()

from Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'G:\Test_Data\2P\220405_L91\220405_L91_stimuli')
from Standard_Stim_Processor import One_Key_Frame_Graphs
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
G16_Dic = Sub_Dic_Generator('G16_2P')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220405_L91\1-003', G16_Dic)
OD_Para = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220405_L91\1-007', OD_Para)
Hue_Para = Sub_Dic_Generator('HueNOrien4',para = 'Default')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220405_L91\1-008', Hue_Para)
#%% Find cell and get cell data.
from Cell_Find_From_Graph import Cell_Find_And_Plot
Morpho_Cells = Cell_Find_And_Plot(day_folder,'Global_Average.tif','Morpho')
from Standard_Cell_Generator import Standard_Cell_Generator
Scg = Standard_Cell_Generator('L91', '220405', day_folder, [1,2,3,4,5,6,7,8],cell_subfolder = r'\_Morpho_Cells')
Scg.Generate_Cells()
from Series_Analyzer.Cell_Frame_PCA import One_Key_PCA
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
Run02_Frame = Pre_Processor(day_folder,'Run002')
Run04_Frame = Pre_Processor(day_folder,'Run004')

before_PCA = One_Key_PCA(day_folder, 'Run002')
After_PCA = One_Key_PCA(day_folder, 'Run004',tag = 'Spon_After')


from Stimulus_Cell_Processor.T_Map_Generator import One_Key_T_Maps
One_Key_T_Maps(day_folder, 'Run007',runtype = 'OD_2P')
