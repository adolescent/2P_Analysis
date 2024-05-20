# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 19:27:57 2022

@author: ZR
"""


from Standard_Aligner import Standard_Aligner
import matplotlib.pyplot as plt
day_folder = r'G:\Test_Data\2P\220407_L85'
Sa = Standard_Aligner(day_folder, [1,2,3,4,5,6,7],final_base = '1-001')
Sa.One_Key_Aligner_No_Affine()

#%% Get stim graphs
from Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'G:\Test_Data\2P\220407_L85\220407_stimuli')
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
from Standard_Stim_Processor import One_Key_Frame_Graphs
G16_Para = Sub_Dic_Generator('G16_2P')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220407_L85\1-002', G16_Para)
OD_Para = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220407_L85\1-006', OD_Para)
Hue_Para = Sub_Dic_Generator('HueNOrien4',para = 'Default')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220407_L85\1-007', Hue_Para)
#%% get cells
from Cell_Find_From_Graph import Cell_Find_And_Plot
Morpho_Cells = Cell_Find_And_Plot(day_folder, 'Global_Average.tif', '_Morpho_Cells')
from Standard_Cell_Generator import Standard_Cell_Generator
Scg = Standard_Cell_Generator('L85', '220407', day_folder, [1,2,3,4,5,6,7],cell_subfolder = r'\_Morpho_Cells')
Scg.Generate_Cells()
#%% Simple PCA
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
from Series_Analyzer.Cell_Frame_PCA import One_Key_PCA
Run01_Frame = Pre_Processor(day_folder,'Run001')
Run03_Frame = Pre_Processor(day_folder,'Run003')
PCA_before = One_Key_PCA(day_folder, 'Run001',tag = 'Spon_Before',start_time=7200)
PCA_after = One_Key_PCA(day_folder, 'Run003',tag = 'Spon_After',start_time=0)
from Stimulus_Cell_Processor.T_Map_Generator import One_Key_T_Maps
One_Key_T_Maps(day_folder, 'Run006',runtype = 'OD_2P')
