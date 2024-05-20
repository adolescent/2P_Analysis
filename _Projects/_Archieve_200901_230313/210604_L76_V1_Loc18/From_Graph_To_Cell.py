# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:01:20 2021

@author: ZR
"""

from My_Wheels.Cell_Find_From_Graph import Cell_Find_From_Mannual
import OS_Tools_Kit as ot

manual_cell = Cell_Find_From_Mannual(r'C:\Users\ZR\Desktop\210604_Cells\_Manual_Cell\Cell_Mask.png',
                                     average_graph_path = r'C:\Users\ZR\Desktop\210604_Cells\_Manual_Cell\Global_Average.tif',
                                     boulder = 5)

from My_Wheels.Standard_Cell_Generator import Standard_Cell_Generator

SCG = Standard_Cell_Generator('L76', '210604', r'K:\Test_Data\2P\210604_L76_2P', list(range(1,17)))
SCG.Generate_Cells()

from My_Wheels.Cell_Processor import Cell_Processor
CP = Cell_Processor(r'K:\Test_Data\2P\210604_L76_2P')
for i in range(CP.cell_num):
    CP.Single_Cell_Plotter(CP.all_cell_names[i])
from My_Wheels.Standard_Parameters.Stim_Name_Tools import Stim_ID_Combiner
OD_Para = Stim_ID_Combiner('OD_2P')
CP.Cell_Response_Maps('Run008',OD_Para,subshape = (3,5))
CP.T_Map_Plot_Core('Run008',[1,3,5,7],[2,4,6,8])
G16_Para = Stim_ID_Combiner('G16_Dirs')
CP.Cell_Response_Maps('Run014',G16_Para,subshape = (3,8))
# Run14 深度不准，对Run16的
CP.Cell_Response_Maps('Run016',G16_Para,subshape = (3,8))
CP.T_Map_Plot_Core('Run016',[1,9],[5,13])
CP.T_Map_Plot_Core('Run016',[3,11],[7,15])
CP.T_Map_Plot_Core('Run016',[1,2,3,4,14,15,16],[6,7,8,9,10,11,12])
CP.T_Map_Plot_Core('Run016',[2,3,4,5,6,7,8],[10,11,12,13,14,15,16])

