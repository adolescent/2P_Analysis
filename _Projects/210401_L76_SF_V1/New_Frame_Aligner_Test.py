# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:11:48 2021

@author: ZR
"""

from My_Wheels.Standard_Aligner import Standard_Aligner


day_folder = r'K:\Test_Data\2P\210401_L76_2P'
runlists = list(range(1,14))
SA = Standard_Aligner(day_folder,runlists,final_base = '1-011')
SA.One_Key_Aligner()

from My_Wheels.Cell_Find_From_Graph import Cell_Find_From_Mannual
cell_dic = Cell_Find_From_Mannual(r'K:\Test_Data\2P\210401_L76_2P\_Manual_Cell\Cell_Mask.png',
                                  r'K:\Test_Data\2P\210401_L76_2P\_Manual_Cell\Global_Average.tif',10)


# Then get cell data
from My_Wheels.Standard_Cell_Generator import Standard_Cell_Generator
SCG = Standard_Cell_Generator('L76', '210401', r'K:\Test_Data\2P\210401_L76_2P', list(range(1,14)))
SCG.Generate_Cells()


