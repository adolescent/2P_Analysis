# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:07:20 2019

@author: ZR

这个程序被用来生成成像的数据帧。输入文件目录和
"""

import numpy as np
import General_Functions.BLK_Tools as BLK_Tools
import General_Functions.OS_Tools as OS_Tools
import General_Functions.Graph_Tools as Graph_Tools
import General_Functions.OI_Sub_Parameters.Standard_Stimulus as Basic

class Condition_Data_Generate(object):
    
    name = r'This class can generate subtraction map after input A/B sets and blk folders'
    
    def __init__(self,data_folder,Sub_Paratemers):
        
        self.all_blk_names = OS_Tools.Path_Control.file_name(data_folder,file_type = '.BLK')#遍历，得到全部的blk名字
        #读取采集data的属性。
        BK_Reader = BLK_Tools.BLK_Reader()
        BK_Reader.BLK_Head_Read(self.all_blk_names[0])
        self.Head_Property = BK_Reader.BLK_Property
        
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
            
    def Condition_Selection(self,target_conditions,blk_names):#输入全部的blk名称和目标的condition序列，返回由全部目标帧组成的图片组。
        
        graph_Num = len(target_conditions)*len(blk_names)#目标的总图片数
        graph_sets = np.zeros(shape = (graph_Num,self.Head_Property['Height'],self.Head_Property['Width']),dtype = np.float64)#第一维度为图片id
        for i in range(len(blk_names)):#循环全部的blk
            current_graph = self.Single_BLK_Process(blk_names[i])#读取当前blk的全部condition
            for j in range(len(target_conditions)):#循环被选中的condition
                graph_sets[i*len(target_conditions)+j,:,:] = current_graph[target_conditions[j]]
        return graph_sets
    
    def Main(self):#主程序，得到全部图片的AB集。
        
        self.Produced_data = {}#这个用来储存全部的待减数据集。
        All_Graph_Sets = self.Sub_Paratemers['All_Graph_Sets']#读取全部的减图集合。
        graph_names = list(All_Graph_Sets.keys())#全部的key，表示所有的图名
        for i in range(len(graph_names)):#对每一幅图
            A_set_id = All_Graph_Sets[graph_names[i]][0]
            B_set_id = All_Graph_Sets[graph_names[i]][1]
            A_sets = self.Condition_Selection(A_set_id,self.all_blk_names)
            B_sets = self.Condition_Selection(B_set_id,self.all_blk_names)
            self.Produced_data[graph_names[i]] = [A_sets,B_sets]
        #OS_Tools.Save_And_Read.save_variable(self.Produced_data,self.save_folder+r'\Produced_data.pkl')#不用保存了，太占空间
        
        
                
if __name__ == '__main__':
    
    data_folder = r'E:\ZR\Data_Temp\191106_L69_OI\Run01_OD8'
    Sub_Paratemers = Basic.OD8_Parameters
    CDG = Condition_Data_Generate(data_folder,Sub_Paratemers)
    CDG.Main()
    Produced_data = CDG.Produced_data
