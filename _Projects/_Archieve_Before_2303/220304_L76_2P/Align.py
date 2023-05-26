# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 19:35:35 2022

@author: ZR
"""

from Standard_Aligner import Standard_Aligner
day_folder = r'G:\Test_Data\2P\220304_L76_2P'
Sa = Standard_Aligner(day_folder, [1,2,3,4,5,6,7,8])
Sa.One_Key_Aligner_No_Affine()
from Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'G:\Test_Data\2P\220304_L76_2P\220304_L76_stimuli')

#%% Get basic stim maps.
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
G16_Paras = Sub_Dic_Generator('G16_2P')
from Standard_Stim_Processor import One_Key_Frame_Graphs
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220304_L76_2P\1-003', G16_Paras)
OD_Paras = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220304_L76_2P\1-007', OD_Paras)
Hue_Para = Sub_Dic_Generator('HueNOrien4',para = 'Default')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220304_L76_2P\1-008', Hue_Para)
#%% Use morpho cell mask
from Cell_Find_From_Graph import Cell_Find_And_Plot
morpho_cells = Cell_Find_And_Plot(r'G:\Test_Data\2P\220304_L76_2P', 
                                  r'Global_Average.tif', 'Morpho')
from Standard_Cell_Generator import Standard_Cell_Generator
Scg = Standard_Cell_Generator('L76', '220304', r'G:\Test_Data\2P\220304_L76_2P', 
                              [1,2,3,4,5,6,7,8],cell_subfolder = r'\Morpho_Cells')
Scg.Generate_Cells()
#%% Test Run01 PCAs.
from Series_Analyzer.Cell_Frame_PCA import One_Key_PCA
comp,info,weight = One_Key_PCA(day_folder, 'Run001',start_time= 7000)
#%% some simple PCA analysis.
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
Run01_Frame = Pre_Processor(day_folder,start_time = 7200)
import matplotlib.pyplot as plt
from Series_Analyzer.Cell_Frame_PCA import Do_PCA,Compoment_Visualize
import OS_Tools_Kit as ot
all_cell_dic = ot.Load_Variable(day_folder,'L76_220304A_All_Cells.ac')
comp,info,weight = Do_PCA(Run01_Frame)
Compoment_Visualize(comp, all_cell_dic, r'G:\Test_Data\2P\220304_L76_2P\_All_Results\PCA_Spon_Before')
Run03_Frame = Pre_Processor(day_folder,'Run003')
comp,info,weight = Do_PCA(Run03_Frame)
Compoment_Visualize(comp, all_cell_dic, r'G:\Test_Data\2P\220304_L76_2P\_All_Results\PCA_Spon_After')






