# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 19:04:57 2022

@author: ZR
"""

from Standard_Aligner import Standard_Aligner
import matplotlib.pyplot as plt 

day_folder = r'G:\Test_Data\2P\220310_L85'

Sa = Standard_Aligner(day_folder, [1,2,3,4,5,6,7],final_base = '1-002')
Sa.One_Key_Aligner_No_Affine()

from Cell_Find_From_Graph import Cell_Find_And_Plot
Cell_Find_And_Plot(day_folder, 'Global_Average.tif','Morpho')
from Standard_Cell_Generator import Standard_Cell_Generator
from Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'G:\Test_Data\2P\220310_L85\220310_L85_stimuli')
Scg = Standard_Cell_Generator('L85', '220310', day_folder, [1,2,3,4,5,6,7],cell_subfolder=r'\Morpho')
Scg.Generate_Cells()
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
Run01_Frame = Pre_Processor(day_folder,'Run001')
from Series_Analyzer.Cell_Frame_PCA import One_Key_PCA
comp_af,info_af,weight_af = One_Key_PCA(day_folder, 'Run003',tag = 'Spon_After')
#%% Get standard stim maps.
from Standard_Stim_Processor import One_Key_Frame_Graphs
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator

G16_Para = Sub_Dic_Generator('G16_2P')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220310_L85\1-002',G16_Para)
OD_Pars = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220310_L85\1-006', OD_Pars)
Hue_Para = Sub_Dic_Generator('HueNOrien4',para = 'Default')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220310_L85\1-007', Hue_Para)

#%% Align Loc2-V2 here.
Sa2 = Standard_Aligner(day_folder, [8],final_base = '1-008')
Sa2.One_Key_Aligner_No_Affine()
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220310_L85\1-008', OD_Pars)
#%% Compare Location with OI Map.

from OI_Graph_Cutter import OI_Graph_Cutter
OI_Graph_Cutter(r'G:\Test_Data\L85_All_OI_Maps\220302_Maps', 
                area_mask_path = day_folder+r'\Mask.png',rotate_angle=97)	
#%% Get Run03 Frames.
Run03_Frame = Pre_Processor(day_folder,'Run003')
