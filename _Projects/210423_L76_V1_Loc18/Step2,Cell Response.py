# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:31:14 2021

@author: ZR
"""

from My_Wheels.Standard_Parameters.Stim_Name_Tools import Stim_ID_Combiner
from My_Wheels.Cell_Processor import Cell_Processor

CP = Cell_Processor(r'K:\Test_Data\2P\210423_L76_2P')
OD_Dic = Stim_ID_Combiner('OD_2P')
CP.Cell_Response_Maps('Run009', OD_Dic,subshape = (3,5))
G16_Dic = Stim_ID_Combiner('G16_Dirs')
