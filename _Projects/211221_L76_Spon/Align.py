# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 19:57:09 2021

@author: ZR
"""

from My_Wheels.Standard_Aligner import Standard_Aligner
from My_Wheels.Stim_Frame_Align import One_Key_Stim_Align


Sa = Standard_Aligner(r'G:\Test_Data\2P\211221_L76_2P', list(range(1,8)))
Sa.One_Key_Aligner_No_Affine()
One_Key_Stim_Align(r'G:\Test_Data\2P\211221_L76_2P\211221_L76_stimuli')


#%% Then calculate the basic frame map.
from My_Wheels.Standard_Stim_Processor import One_Key_Frame_Graphs
from My_Wheels.Standard_Parameters.Sub_Graph_Dics import Sub_Dic_Generator
G16_Para = Sub_Dic_Generator('G16_2P')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\211221_L76_2P\1-002',G16_Para)
OD_Para = Sub_Dic_Generator('OD_2P')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\211221_L76_2P\1-006',OD_Para)
Hue_Para = Sub_Dic_Generator('HueNOrien4',para = 'Default')
One_Key_Frame_Graphs(r'G:\Test_Data\2P\211221_L76_2P\1-007',Hue_Para)

#%% Get Morpho cells
from Cell_Find_From_Graph import Cell_Find_And_Plot
morpho_cells = Cell_Find_And_Plot(r'G:\Test_Data\2P\211221_L76_2P', 'Global_Average.tif',
                                  'Morpho',find_thres = 1.25)

#%% Use Manual Cell
from Cell_Find_From_Graph import Cell_Find_From_Mannual
manual_cell = Cell_Find_From_Mannual(r'G:\Test_Data\2P\211221_L76_2P\_Manual_Cells\Cell_Mask.png'
                                     ,average_graph_path=r'G:\Test_Data\2P\211221_L76_2P\_Manual_Cells\Global_Average.tif')

#%% Generate cells
from Standard_Cell_Generator import Standard_Cell_Generator
Scg = Standard_Cell_Generator('L76', '211221', r'G:\Test_Data\2P\211221_L76_2P',
                              list(range(1,8)))
Scg.Generate_Cells()

#%% Test PCA
from Series_Analyzer.Cell_Frame_PCA import One_Key_PCA,Do_PCA,Compoment_Visualize
from Series_Analyzer.Spontaneous_Preprocessing import Pre_Processor
from Series_Analyzer.Cell_Activity_Evaluator import Pre_Processed_Data_Count
import matplotlib.pyplot as plt
import OS_Tools_Kit as ot

day_folder = r'G:\Test_Data\2P\211221_L76_2P'
save_folder = r'G:\Test_Data\2P\211221_L76_2P\_All_Results\PCA_Before'
all_cell_dic = ot.Load_Variable(r'G:\Test_Data\2P\211221_L76_2P\L76_211221A_All_Cells.ac')
Run01_frame = Pre_Processor(day_folder,passed_band = (0.005,0.3),order = 7,start_time = 10800)
count,_ = Pre_Processed_Data_Count(Run01_frame)
plt.plot(count.mean(0))
comp,info,weights = Do_PCA(Run01_frame)
Compoment_Visualize(comp, all_cell_dic, day_folder+r'\_All_Results\PCA_Before')
ot.Save_Variable(save_folder, 'PCA_info', info)
ot.Save_Variable(save_folder, 'Fitted_Weights', weights)
ot.Save_Variable(r'G:\_Pre_Processed_Data\211221_L76_Loc14_0.005-0.3', 'Run01_302cells_10800s-All_Spon_Before',Run01_frame)

save_folder_after = r'G:\Test_Data\2P\211221_L76_2P\_All_Results\PCA_After'
Run03_frame = Pre_Processor(day_folder,'Run003',passed_band = (0.005,0.3),order = 7,start_time = 0)
count,_ = Pre_Processed_Data_Count(Run03_frame)
plt.plot(count.mean(0))
comp,info,weights = Do_PCA(Run03_frame)
Compoment_Visualize(comp, all_cell_dic, day_folder+r'\_All_Results\PCA_After')
ot.Save_Variable(save_folder_after, 'PCA_info', info)
ot.Save_Variable(save_folder_after, 'Fitted_Weights', weights)
ot.Save_Variable(r'G:\_Pre_Processed_Data\211221_L76_Loc14_0.005-0.3', 'Run03_302cells_0s-All_Spon_Before',Run03_frame)



