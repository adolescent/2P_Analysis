# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:55:26 2019

@author: ZR

This function provide multiple tools for graph processing.
Usually used filter & Normalization are contained.

F1：clip
F2：基于图片文件的平均
F3：归一化
F4：写图片
"""
import numpy as np
import cv2


class Graph_Processing(object):
    
    name = r'Small tools used for graph processing'
#%% F1 clip函数。注意这个函数只能指定输出data type，需要确保输入的data type是正确的，否则可能会出想象外的bug
    def Graph_Clip(graph,std_Num,Formation = 'f8'):
        
        '''This function will clip graph in mean+-std, Out put formation can be defined here. float64 output is default.'''
        #定义数据类型
        if (Formation != 'f8' and Formation !='u2' and Formation != 'u1'):#f8:float64,u1:uint8;u2,uint16.
            
            raise ValueError('Data type not understood!')
        else:
            dt = np.dtype(Formation)
        #接下来初始化
        data_mean = graph.mean()
        data_std = graph.std()
        lower_level = data_mean-std_Num*data_std#数据的下界限
        upper_level = data_mean+std_Num*data_std
        clipped_graph = np.clip(graph,lower_level,upper_level).astype(dt)
        
        return clipped_graph
#%% F2 用于平均多幅图，只用于平均文件，且需要输入的图片长宽、位深度匹配，否则会报错。  
    def Graph_File_Average(file_names,Formation = 'f8'):
        #输入的必须是文件名组成的list
        graph_Nums = len(file_names)
        temp_graph = cv2.imread(file_names[0],-1)
        averaged_frame = np.zeros(shape = np.shape(temp_graph),dtype = np.float64)
        for i in range(graph_Nums):
            temp_graph = cv2.imread(file_names[i],-1).astype('f8')
            averaged_frame += (temp_graph/graph_Nums)
        return averaged_frame.astype(Formation)#输出格式和输入一样。
#%% F3 输出最大-最小拉伸过的图，一般用于处理clip后的结果。   
    def Graph_Normalization(graph,bit = 'u1'):
        
        max_value = graph.max()
        min_value = graph.min()
        normalized_graph = (graph-min_value)/(max_value-min_value)#归一化到1-0
        if bit == 'u1':
            return (normalized_graph*255).astype(bit)
        elif bit == 'u2':
            return (normalized_graph*65535).astype(bit)
        else:
            return normalized_graph
            print('Attention Here, 0~1 graph data produced here.')
        
#%% F4 写图片，写入之前show一下，如果wait_time设置成0则跳过show的步骤。      
    def Write_Graph(path,graph,name,wait_time = 2500):
        
        if wait_time != 0: #等待时间不为0则show
            cv2.imshow(name,graph)
            cv2.waitKey(wait_time)
            cv2.destroyAllWindows()
        cv2.imwrite(path+r'\\'+name+'.tif',graph)
        
        
        
        
#%% Test functions below.        
if __name__ == '__main__':
    
    #test = Graph_Processing.Graph_File_Average(global_tif_name[20:50])
    print('Test Run Ended.\n')
