# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 18:57:53 2022

@author: ZR
"""

from Standard_Aligner import Standard_Aligner
day_folder = r'G:\Test_Data\2P\220326_L76'

Sa = Standard_Aligner(day_folder, [1,2,3,4,5,6,7],final_base = '1-003')
Sa.One_Key_Aligner_No_Affine()
Sa.Get_Final_Average()
#%% get test 
import OS_Tools_Kit as ot
import Graph_Operation_Kit as gt
all_tif_name = ot.Get_File_Name(r'G:\Test_Data\2P\220326_L76\1-001\Results\Final_Aligned_Frames')
part_tif_name = all_tif_name[5000:]
avr = gt.Average_From_File(part_tif_name)
clipped_avr = gt.Clip_And_Normalize(avr,5)
gt.Show_Graph(clipped_avr, 'Last_Part', r'G:\Test_Data\2P\220326_L76\1-001\Results')
#%% Get basic maps.
from Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'G:\Test_Data\2P\220326_L76\220326_stimuli')
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
G16_Para = Sub_Dic_Generator('G16_2P')
from Standard_Stim_Processor import One_Key_Frame_Graphs
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220326_L76\1-002', G16_Para)
OD_Para = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220326_L76\1-006', OD_Para)
Hue_Para = Sub_Dic_Generator('HueNOrien4',para = 'Default')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220326_L76\1-007', Hue_Para)
#%% Find cell for Run01
from Cell_Find_From_Graph import Cell_Find_And_Plot
Run01_Cell = Cell_Find_And_Plot(day_folder, 'Run01_avr.tif', 'Morpho_Run01')
Other_Cell = Cell_Find_And_Plot(day_folder, 'Run03_avr.tif', 'Morpho_Cells')
from Standard_Cell_Generator import Standard_Cell_Generator
Scg = Standard_Cell_Generator('L76', '220326', day_folder, [2,3,4,5,6,7],cell_subfolder = r'\_Morpho_Cells')
Scg.Generate_Cells()
from Series_Analyzer.Cell_Frame_PCA import One_Key_PCA
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
Run01_Before = Pre_Processor(day_folder,'Run001')
PCA_before = One_Key_PCA(day_folder, 'Run001',tag = 'Spon_Before',start_time = 6000)
Run03_after = Pre_Processor(day_folder,'Run003')
PCA_after = One_Key_PCA(day_folder, 'Run003',tag = 'Spon_After',start_time = 0)
from Stimulus_Cell_Processor.T_Map_Generator import One_Key_T_Maps
One_Key_T_Maps(day_folder, 'Run006',runtype = 'OD_2P')
