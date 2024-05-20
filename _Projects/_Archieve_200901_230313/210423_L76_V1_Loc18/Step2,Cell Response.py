# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:31:14 2021

@author: ZR
"""

from My_Wheels.Standard_Parameters.Stim_Name_Tools import Stim_ID_Combiner
from My_Wheels.Cell_Processor import Cell_Processor
import cv2
import OS_Tools_Kit as ot

#average_graph = cv2.imread(r'K:\Test_Data\2P\210423_L76_2P\_Manual_Cell\Global_Average.tif',-1)
CP = Cell_Processor(r'K:\Test_Data\2P\210423_L76_2P')
OD_Dic = Stim_ID_Combiner('OD_2P')
CP.Cell_Response_Maps('Run009', OD_Dic,subshape = (3,5))
G16_Dic = Stim_ID_Combiner('G16_Dirs')
CP.Cell_Response_Maps('Run014', G16_Dic,subshape = (3,8))
Hue11_Dic = Stim_ID_Combiner('HueNOrien4_Color',para_dic={'Hue':['Red0.6','Red0.5','Red0.4','Red0.3','Red0.2','Yellow','Green','Cyan','Blue','Purple','White']})
CP.Cell_Response_Maps('Run016', Hue11_Dic)
Hue_11_SC_Dic = Stim_ID_Combiner('HueNOrien4_SC',para_dic ={'Hue':['Red0.6','Red0.5','Red0.4','Red0.3','Red0.2','Yellow','Green','Cyan','Blue','Purple','White']} )
CP.Cell_Response_Maps('Run016',Hue_11_SC_Dic,subshape = (6,11),figsize = (25,20))
All_Black = CP.Black_Cell_Identifier(['Run009','Run014','Run016'])
ot.Save_Variable(r'K:\Test_Data\2P\210423_L76_2P', '_All_Black', All_Black)
#%% Plot all cells
for i in range(CP.cell_num):
    CP.Single_Cell_Plotter(CP.all_cell_names[i],show_time = 0)
all_black_cell_name = list(All_Black.keys())
CP.Part_Cell_Plotter(all_black_cell_name)
CP.Part_Cell_Plotter(all_black_cell_name,mode = 'fill')
black_cell_num = len(All_Black)
#%% let's see different run seperately.
OD_neg_cell = []
G16_neg_cell = []
Hue_neg_cell= []
for i in range(black_cell_num):
    c_cell_name = all_black_cell_name[i]
    if 'Run009' in All_Black[c_cell_name]:
        OD_neg_cell.append(c_cell_name)
    if 'Run014' in All_Black[c_cell_name]:
        G16_neg_cell.append(c_cell_name)
    if 'Run016' in All_Black[c_cell_name]:
        Hue_neg_cell.append(c_cell_name)
CP.Part_Cell_Plotter(OD_neg_cell)
CP.Part_Cell_Plotter(OD_neg_cell,mode = 'circle')
CP.Part_Cell_Plotter(G16_neg_cell)
CP.Part_Cell_Plotter(G16_neg_cell,mode = 'circle')
CP.Part_Cell_Plotter(Hue_neg_cell)
CP.Part_Cell_Plotter(Hue_neg_cell,mode = 'circle')
#%% get F value distribution
CP.Part_Cell_F_Disp(all_black_cell_name,graph_name = 'All_Black_Cells_F')
CP.Part_Cell_F_Disp(OD_neg_cell,graph_name = 'OD_Black_Cells_F',bins = 5)
CP.Part_Cell_F_Disp(G16_neg_cell,graph_name = 'G16_Black_Cells_F',bins = 5)
CP.Part_Cell_F_Disp(Hue_neg_cell,graph_name = 'Hue_Black_Cells_F',bins = 10)