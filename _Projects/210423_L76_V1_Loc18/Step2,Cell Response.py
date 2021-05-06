# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:31:14 2021

@author: ZR
"""

from My_Wheels.Standard_Parameters.Stim_Name_Tools import Stim_ID_Combiner
from My_Wheels.Cell_Processor import Cell_Processor
import cv2

average_graph = cv2.imread(r'K:\Test_Data\2P\210423_L76_2P\_Manual_Cell\Global_Average.tif',-1)
CP = Cell_Processor(r'K:\Test_Data\2P\210423_L76_2P',average_graph)
OD_Dic = Stim_ID_Combiner('OD_2P')
CP.Cell_Response_Maps('Run009', OD_Dic,subshape = (3,5))
G16_Dic = Stim_ID_Combiner('G16_Dirs')
CP.Cell_Response_Maps('Run014', G16_Dic,subshape = (3,8))
Hue11_Dic = Stim_ID_Combiner('HueNOrien4_Color',para_dic={'Hue':['Red0.6','Red0.5','Red0.4','Red0.3','Red0.2','Yellow','Green','Cyan','Blue','Purple','White']})
CP.Cell_Response_Maps('Run016', Hue11_Dic)
Hue_11_SC_Dic = Stim_ID_Combiner('HueNOrien4_SC',para_dic ={'Hue':['Red0.6','Red0.5','Red0.4','Red0.3','Red0.2','Yellow','Green','Cyan','Blue','Purple','White']} )
CP.Cell_Response_Maps('Run016',Hue_11_SC_Dic,subshape = (6,11),figsize = (25,20))

