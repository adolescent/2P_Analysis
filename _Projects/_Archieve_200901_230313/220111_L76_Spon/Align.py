# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 20:08:35 2022

@author: ZR
"""

from Standard_Aligner import Standard_Aligner
import matplotlib.pyplot as plt
day_folder = r'G:\Test_Data\2P\220111_L76_2P'

Sa = Standard_Aligner(day_folder, [1,2,3,4,5,6,7])
Sa.One_Key_Aligner_No_Affine()

#%% Stim frame align
from Stim_Frame_Align import One_Key_Stim_Align
One_Key_Stim_Align(r'G:\Test_Data\2P\220111_L76_2P\220111_L76_stimuli')

from Standard_Stim_Processor import One_Key_Frame_Graphs
from Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
G16_Para = Sub_Dic_Generator('G16_2P')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220111_L76_2P\1-002', G16_Para)
OD_Para = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220111_L76_2P\1-006', OD_Para)
Hue_Para = Sub_Dic_Generator('HueNOrien4',para = 'Default')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\220111_L76_2P\1-007', Hue_Para)

#%%
from Cell_Find_From_Graph import Cell_Find_And_Plot
Morpho_Cell = Cell_Find_And_Plot(r'G:\Test_Data\2P\220111_L76_2P\_Morpho_Cell','Global_Average.tif','Morpho',find_thres = 1.5)
from Standard_Cell_Generator import Standard_Cell_Generator
Scg = Standard_Cell_Generator('L76', '220111', r'G:\Test_Data\2P\220111_L76_2P', [1,2,3,4,5,6,7],
                              cell_subfolder = r'\_Morpho_Cell')
Scg.Generate_Cells()

from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
Run01_all = Pre_Processor(day_folder,runname = 'Run001',start_time = 0,stop_time=99999)
Run01_12000 = Run01_all.iloc[:,12000:]
from Series_Analyzer.Cell_Frame_PCA import Do_PCA,Compoment_Visualize
import OS_Tools_Kit as ot
all_cell_dic = ot.Load_Variable(day_folder,r'L76_220111A_All_Cells.ac')
comp,info,weights = Do_PCA(Run01_12000)
_ = Compoment_Visualize(comp, all_cell_dic, r'G:\Test_Data\2P\220111_L76_2P\_All_Results')
ot.Save_Variable(r'G:\Test_Data\2P\220111_L76_2P\_All_Results\PCA_Before',
                 'PCA_Components',comp)
ot.Save_Variable(r'G:\Test_Data\2P\220111_L76_2P\_All_Results\PCA_Before',
                 'PCA_Information',info)
ot.Save_Variable(r'G:\Test_Data\2P\220111_L76_2P\_All_Results\PCA_Before',
                 'PCA_weights',weights)


Run03_all = Pre_Processor(day_folder,runname = 'Run003',start_time = 0,stop_time=99999)
comp,info,weights = Do_PCA(Run03_all)
_ = Compoment_Visualize(comp, all_cell_dic, r'G:\Test_Data\2P\220111_L76_2P\_All_Results')
ot.Save_Variable(r'G:\Test_Data\2P\220111_L76_2P\_All_Results\PCA_After',
                 'PCA_Components',comp)
ot.Save_Variable(r'G:\Test_Data\2P\220111_L76_2P\_All_Results\PCA_After',
                 'PCA_Information',info)
ot.Save_Variable(r'G:\Test_Data\2P\220111_L76_2P\_All_Results\PCA_After',
                 'PCA_weights',weights)


#%% Manual Cells
from Cell_Find_From_Graph import Cell_Find_From_Mannual
manual_cell = Cell_Find_From_Mannual(r'G:\Test_Data\2P\220111_L76_2P\_Manual_Cells\Cell_Mask.png',
                                     r'G:\Test_Data\2P\220111_L76_2P\_Manual_Cells\Global_Average.tif')
Scg = Standard_Cell_Generator('L76', '220111', r'G:\Test_Data\2P\220111_L76_2P', [1,2,3,4,5,6,7])
Scg.Generate_Cells()



