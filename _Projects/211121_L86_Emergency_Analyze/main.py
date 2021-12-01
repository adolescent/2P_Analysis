# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:57:53 2021

@author: ZR
"""

from My_Wheels.Standard_Aligner import Standard_Aligner
from Stim_Frame_Align import One_Key_Stim_Align
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator

day_folder = r'G:\Test_Data\2P\211121_L86_2P'
Sa = Standard_Aligner(day_folder, [3,9,12],final_base = '1-009')
Sa.One_Key_Aligner()	
One_Key_Stim_Align(r'G:\Test_Data\2P\211121_L86_2P\211121_L86LL_stimuli')

#%% Get cell from morphology
from My_Wheels.Cell_Find_From_Graph import Cell_Find_And_Plot
morpho_cells = Cell_Find_And_Plot(r'G:\Test_Data\2P\211121_L86_2P\1-003\Results',
                                  'Global_Average.tif', 'Morpho',find_thres = 1.5)
from Standard_Cell_Generator import Standard_Cell_Generator
Scg = Standard_Cell_Generator('L86', '211121', r'G:\Test_Data\2P\211121_L86_2P', [3,9,12]
                              ,cell_subfolder=r'\Morpho_Cell')
Scg.Generate_Cells()
#%% Get Color Frame Dics
Run09_Sub_Dic = {}
Run09_Sub_Dic['All_Color_HV'] = [[3,7,11,15,19,23,27,31,33,39,41,47],[1,5,9,13,17,21,25,29,33,37,41,45]]
Run09_Sub_Dic['All_Color_AO'] = [[4,8,12,16,20,24,28,32,36,40,44,48],[2,6,10,14,18,22,26,30,34,38,42,46]]
Run09_Sub_Dic['Red-White'] = [list(range(1,9)),list(range(33,41))]
Run09_Sub_Dic['Green-White'] = [list(range(9,17)),list(range(33,41))]
Run09_Sub_Dic['Blue-White'] = [list(range(25,33)),list(range(33,41))]
from Standard_Stim_Processor import One_Key_Frame_Graphs
One_Key_Frame_Graphs(r'G:\Test_Data\2P\211121_L86_2P\1-009', Run09_Sub_Dic)
from My_Wheels.Stimulus_Cell_Processor.T_Map_Generator import One_Key_T_Maps
t_maps = One_Key_T_Maps(day_folder, 'Run009',runtype = Run09_Sub_Dic)


#%% Test Align without affine.
