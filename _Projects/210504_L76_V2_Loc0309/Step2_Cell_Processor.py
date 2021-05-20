# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:20:09 2021


@author: ZR
Process cell data here.
"""

from My_Wheels.Cell_Processor import Cell_Processor
import OS_Tools_Kit as ot
from My_Wheels.Standard_Parameters.Stim_Name_Tools import Stim_ID_Combiner


#average_graph = cv2.imread(r'K:\Test_Data\2P\210504_L76_2P\_Manual_Cell\Global_Average.tif',-1)
CP = Cell_Processor(r'K:\Test_Data\2P\210504_L76_2P')
G16_Dic = Stim_ID_Combiner('G16_Dirs')
CP.Cell_Response_Maps('Run013', G16_Dic,subshape = (3,8))
G16_Rad = Stim_ID_Combiner('G16_Radar')
CP.Radar_Maps('Run013', G16_Rad)
OD_Dic = Stim_ID_Combiner('OD_2P')
CP.Cell_Response_Maps('Run010', OD_Dic,subshape = (3,5))
OD_Rad = Stim_ID_Combiner('OD_2P_Radar')
CP.Radar_Maps('Run010', OD_Rad,bais_angle=22.5)
S3D8_Dic = Stim_ID_Combiner('Shape3Dir8_Single')
CP.Cell_Response_Maps('Run015', S3D8_Dic,subshape = (4,8))
S3D8_General = Stim_ID_Combiner('Shape3Dir8_General')
CP.Cell_Response_Maps('Run015', S3D8_General)
H7O4_SC = Stim_ID_Combiner('HueNOrien4_SC',{'Hue':['Red','Yellow','Green','Cyan','Blue','Purple','White']})
CP.Cell_Response_Maps('Run016', H7O4_SC,subshape = (6,7),figsize = (20,20))
All_Black_Cells = CP.Black_Cell_Identifier(['Run010','Run013','Run015','Run016'])
ot.Save_Variable(r'K:\Test_Data\2P\210504_L76_2P', '_All_Black', All_Black_Cells)
#%% Then statictic all single condition blacks.
for i in range(CP.cell_num):
    CP.Single_Cell_Plotter(CP.all_cell_names[i],show_time = 0)
black_cell_num = len(All_Black_Cells)
all_black_cell_name = list(All_Black_Cells.keys())
OD_neg_cell = []
G16_neg_cell = []
Hue_neg_cell= []
Shape_neg_cell = []
for i in range(black_cell_num):
    c_cell_name = all_black_cell_name[i]
    if 'Run010' in All_Black_Cells[c_cell_name]:
        OD_neg_cell.append(c_cell_name)
    if 'Run013' in All_Black_Cells[c_cell_name]:
        G16_neg_cell.append(c_cell_name)
    if 'Run016' in All_Black_Cells[c_cell_name]:
        Hue_neg_cell.append(c_cell_name)
    if 'Run015' in All_Black_Cells[c_cell_name]:
        Shape_neg_cell.append(c_cell_name)
        
CP.Part_Cell_Plotter(all_black_cell_name)
CP.Part_Cell_Plotter(all_black_cell_name,mode ='fill')
CP.Part_Cell_Plotter(OD_neg_cell)
CP.Part_Cell_Plotter(OD_neg_cell,mode = 'fill')
CP.Part_Cell_Plotter(G16_neg_cell)
CP.Part_Cell_Plotter(G16_neg_cell,mode = 'fill')
CP.Part_Cell_Plotter(Hue_neg_cell)
CP.Part_Cell_Plotter(Hue_neg_cell,mode = 'fill')
CP.Part_Cell_Plotter(Shape_neg_cell)
CP.Part_Cell_Plotter(Shape_neg_cell,mode = 'fill')

#%% Black Cell F value
CP.Part_Cell_F_Disp(all_black_cell_name,'All_Blacks_F')
CP.Part_Cell_F_Disp(OD_neg_cell,'OD_Blacks_F',bins = 5)
CP.Part_Cell_F_Disp(G16_neg_cell,'G16_Blacks_F',bins = 10)
CP.Part_Cell_F_Disp(Hue_neg_cell,'Hue_Blacks_F',bins = 10)
CP.Part_Cell_F_Disp(Shape_neg_cell,'Shape_Blacks_F',bins = 5)