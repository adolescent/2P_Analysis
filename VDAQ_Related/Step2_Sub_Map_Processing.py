# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:07:20 2019

@author: ZR

这个程序被用来生成成像的减图，参照帧、刺激减法、都需要提前指定。
"""

import numpy as np
import General_Functions.BLK_Tools as BLK_Tools
import General_Functions.OS_Tools as OS_Tools
import General_Functions.Graph_Tools as Graph_Tools
import General_Functions.OI_Sub_Parameters.Standard_Stimulus as Basic

class Sub_Map_Generate(object):
    
    name = r'This class can generate subtraction map after input A/B sets and blk folders'
    
    def __init__(self,data_folder,Sub_Paratemers):
        
        self.all_blk_names = OS_Tools.Path_Control.file_name(data_folder,file_type = '.BLK')
        self.save_folder = data_folder+r'\Results'
        OS_Tools.Path_Control.mkdir(self.save_folder)
        self.Sub_Paratemers = Sub_Paratemers
        
    def Single_BLK_Process(self,blk_name):#输入一个blk的名字，返回sub过的blk，每个cond是一张图。
        BK_Reader = BLK_Tools.BLK_Reader()
        Current_BLK = BK_Reader.Direct_BLK_Read(blk_name)#这里已经存了一个字典，key是id，value是16张图。
        ref_frame_id =self.Sub_Paratemers['Ref_Frame']
        data_frame_id = self.Sub_Paratemers['Data_Frame']
        
        processed_blk = {}#定义空矩阵，填入每个condition对应的一副图。
        condition_IDs = list(Current_BLK.keys())
        for i in range(len(condition_IDs)):
            current_condition = Current_BLK[condition_IDs[i]]#共16幅图，对应1-16帧。
            ref_frame = Graph_Tools.Graph_Processing.Graph_Matrix_Average(current_condition[ref_frame_id,:,:],0)
            data_frame = Graph_Tools.Graph_Processing.Graph_Matrix_Average(current_condition[data_frame_id,:,:],0)
            condition_graph = data_frame-ref_frame
            processed_blk[condition_IDs[i]] = condition_graph
        return processed_blk
            
    
            
        
        
        
        
    
        
        
if __name__ == '__main__':
    
    data_folder = r'E:\ZR\Data_Temp\180629_L63_OI_Run01_G8_Test'
    Sub_Paratemers = Basic.G8_Parameters
    SMG = Sub_Map_Generate(data_folder,Sub_Paratemers)
    a = SMG.all_blk_names
    test = SMG.Single_BLK_Process(a[0])
