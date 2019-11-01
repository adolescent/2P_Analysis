# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:07:20 2019

@author: ZR

这个程序被用来生成成像的减图，参照帧、刺激减法、都需要提前指定。
"""

import numpy as np
import General_Functions.OS_Tools as OS_Tools

class Sub_Map_Generate(object):
    
    name = r'This class can generate subtraction map after input A/B sets and blk folders'
    
    def __init__(self,data_folder,SubTraction_Parameters):
        
        self.all_blk_names = OS_Tools.Path_Control.file_name(data_folder,file_type = '.BLK')
        
        
        
        
    
        
        
if __name__ == '__main__':
    
    data_folder = r'E:\ZR\Data_Temp\180629_L63_OI_Run01_G8_Test'
    SMG = Sub_Map_Generate(data_folder,4)
    a = SMG.all_blk_names
    