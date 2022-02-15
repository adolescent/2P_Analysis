# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 19:44:41 2022

@author: ZR
"""

from Standard_Aligner import Standard_Aligner
day_folder = r'G:\Test_Data\2P\220211_L76_2P'

Sa = Standard_Aligner(day_folder, [1,2,3,4,5])
Sa.One_Key_Aligner_No_Affine()

from Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'G:\Test_Data\2P\220211_L76_2P\220211_L76_2P_stimuli')
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
G16_Para = Sub_Dic_Generator('G16_2P')
from Standard_Stim_Processor import One_Key_Frame_Graphs
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220211_L76_2P\1-002',G16_Para)
Shape_Para = {}
Shape_Para['H-V'] = [[1,5],[3,7]]
Shape_Para['A-O'] = [[2,6],[4,8]]
Shape_Para['Circle-Bar'] = [[25,26,27,28,29,30,31,32],[1,2,3,4,5,6,7,8]]
Shape_Para['Circle-Triangle'] = [[25,26,27,28,29,30,31,32],[17,18,19,20,21,22,23,24]]
Shape_Para['Tirangle-Bar'] = [[17,18,19,20,21,22,23,24],[1,2,3,4,5,6,7,8	]]
Shape_Para['RD_L-R'] = [[12,13,14],[9,10,16]]
Shape_Para['RD_U-D'] = [[10,11,12],[14,15,16]]
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220211_L76_2P\1-004',Shape_Para)
Hue_Para = Sub_Dic_Generator('HueNOrien4',para = 'Default')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220211_L76_2P\1-005',Hue_Para)

#%% cells from morpho
from Cell_Find_From_Graph import Cell_Find_And_Plot
Cell_Find_And_Plot(r'G:\Test_Data\2P\220211_L76_2P', 'Global_Average.tif', 'Morpho')
from Standard_Cell_Generator import Standard_Cell_Generator
Scg = Standard_Cell_Generator('L76', '220211', r'G:\Test_Data\2P\220211_L76_2P',
                              [1,2,3,4,5],cell_subfolder=r'\Morpho_Cells')
Scg.Generate_Cells()
#%% OneKey PCA
from Series_Analyzer.Cell_Frame_PCA import One_Key_PCA
import OS_Tools_Kit as ot
One_Key_PCA(r'G:\Test_Data\2P\220211_L76_2P','Run001',start_time= 7000)
One_Key_PCA(r'G:\Test_Data\2P\220211_L76_2P','Run003',start_time= 0,tag = 'Spon_After')
