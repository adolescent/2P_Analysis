# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:20:09 2021


@author: ZR
Process cell data here.
"""

from My_Wheels.Cell_Processor import Cell_Processor
import cv2
from My_Wheels.Standard_Parameters.Stim_Name_Tools import Stim_ID_Combiner

average_graph = cv2.imread(r'K:\Test_Data\2P\210504_L76_2P\_Manual_Cell\Global_Average.tif',-1)
CP = Cell_Processor(r'K:\Test_Data\2P\210504_L76_2P',average_graph)
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


