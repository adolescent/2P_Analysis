# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 19:17:31 2022

@author: ZR
"""

from Standard_Aligner import Standard_Aligner

day_folder = r'G:\Test_Data\2P\220324_L85'
Sa = Standard_Aligner(day_folder, [1,2,3,4,5,6,7],final_base = '1-003')
Sa.One_Key_Aligner_No_Affine()

from Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'G:\Test_Data\2P\220324_L85\220324_L85_stimuli')


from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
G16_Para = Sub_Dic_Generator('G16_2P')
from Standard_Stim_Processor import One_Key_Frame_Graphs
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220324_L85\1-002', G16_Para)
OD_Para = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220324_L85\1-006', OD_Para)
Hue_Para = Sub_Dic_Generator('HueNOrien4',para = 'Default')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220324_L85\1-007', Hue_Para)
#%% Get all cell data here.
from Cell_Find_From_Graph import Cell_Find_And_Plot
morpho_cells = Cell_Find_And_Plot(r'G:\Test_Data\2P\220324_L85', r'Global_Average.tif','Morpho')
from Standard_Cell_Generator import Standard_Cell_Generator
Scg = Standard_Cell_Generator('L85', '220324', day_folder, [1,2,3,4,5,6,7],cell_subfolder = r'\Morpho')
Scg.Generate_Cells()

#%% Get simple Spon analysis.
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
from Series_Analyzer.Cell_Frame_PCA import One_Key_PCA


Run01_Frame_Whole = Pre_Processor(day_folder,'Run001')
Run03_Frame_Whole = Pre_Processor(day_folder,'Run003')
before_PCA = One_Key_PCA(day_folder, 'Run001',start_time = 7000)
after_PCA = One_Key_PCA(day_folder, 'Run003',tag = 'Spon_After')

from Stimulus_Cell_Processor.T_Map_Generator import One_Key_T_Maps
OD_tmap = One_Key_T_Maps(day_folder, 'Run006')
G16_tmap = One_Key_T_Maps(day_folder, 'Run002',runtype='G16_2P')
