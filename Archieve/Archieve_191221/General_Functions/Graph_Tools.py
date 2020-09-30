# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:55:26 2019

@author: ZR

This function provide multiple tools for graph processing.
Usually used filter & Normalization are contained.

class Graph_Processing:
    F1：clip
    F2：基于图片文件的平均
    F4：归一化
    F5：写图片

class Alignment:
    主要用主函数，输入模板和要对齐的帧，并可制定对齐范围；输出对齐后的图，边界用中位数填充了
    

"""
import numpy as np
import cv2


class Graph_Processing(object):
    name = r'Small tools used for graph processing'
    # F1 clip函数。注意这个函数只能指定输出data type，需要确保输入的data type是正确的，否则可能会出想象外的bug
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
# F2 用于平均多幅图，只用于平均文件，且需要输入的图片长宽、位深度匹配，否则会报错。  
    def Graph_File_Average(file_names,Formation = 'f8'):
        #输入的必须是文件名组成的list
        graph_Nums = len(file_names)
        temp_graph = cv2.imread(file_names[0],-1)
        averaged_frame = np.zeros(shape = np.shape(temp_graph),dtype = np.float64)
        for i in range(graph_Nums):
            temp_graph = cv2.imread(file_names[i],-1).astype('f8')
            averaged_frame += (temp_graph/graph_Nums)
        return averaged_frame.astype(Formation)#输出格式和输入一样。
# F3 用于平均输入矩阵里面的图片。
    def Graph_Matrix_Average(graph_matrix,number_axis,Formation = 'f8'):#number_axis是：输入矩阵中的不同帧id所在的维度。
        graph_matrix = graph_matrix.astype('f8')
        averaged_graph = np.mean(graph_matrix,axis = number_axis).astype(Formation)
        return averaged_graph
        
# F4 输出最大-最小拉伸过的图，一般用于处理clip后的结果。   
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
        
# F5 写图片，写入之前show一下，如果wait_time设置成0则跳过show的步骤。      
    def Write_Graph(path,graph,name,graph_formation = '.tif',wait_time = 2500):
        
        if wait_time != 0: #等待时间不为0则show
            cv2.imshow(name,graph)
            cv2.waitKey(wait_time)
            cv2.destroyAllWindows()
        cv2.imwrite(path+r'\\'+name+graph_formation,graph)
        
# F6 绘制彩图，输入一副正负值的图片，归一化并输出
    def Graph_Colorization(input_graph,bit = 'u2',show_time = 2500,positive_color = [0,1,0],negative_color = [0,0,1]):#BGR顺序注意
        
        graph_height,graph_width = np.shape(input_graph)
        colorized_graph = np.zeros(shape = (graph_height,graph_width,3),dtype ='f8')
        
        input_graph = input_graph.astype('f8')#先都转成float64，防止舍入造成问题
        symbol = input_graph>0#正负的标注
        positive_part = input_graph*symbol#正的部分
        negative_part = input_graph*(symbol-1)#负的部分，主要这里已经去掉了符号，都是正的。
        for i in range(3):#三个通道
            colorized_graph[:,:,i] = colorized_graph[:,:,i]+positive_part*positive_color[i]+negative_part*negative_color[i]
            
        #接下来进行归一化，转成目标的位深度。
        norm_colorized_graph = (colorized_graph - colorized_graph.min())/(colorized_graph.max()-colorized_graph.min())#归一化至0-1的范围
        if bit == 'u1':
            colorized_graph = (norm_colorized_graph*255).astype('u1')
        elif bit == 'u2':
            colorized_graph = (norm_colorized_graph*65535).astype('u2')
        if show_time != 0:
            cv2.imshow('Colorized_graph',colorized_graph)
            cv2.waitKey(show_time)
            cv2.destroyAllWindows()
        return colorized_graph
        
    
    
class Alignment(object):
    
    name = r'Tools used for Graph Align.This class will input Graphs, then output aligned graph.'   
    def __init__(self,target,tample,temple_boulder = 20,align_range = 20):#target是要对齐的图像；tample是目标模板        
        self.baised_graph = self.main(target,tample,tample_boulder = 20,align_range = 20)

    
    def boulder_cut(self,graph,boulder):#切割边界，由于双光在xy方向的抖动是一样的，所以只切割同一个大小的了
        length,width = np.shape(graph)
        cutted_graph = graph[boulder:(length-boulder),boulder:(width-boulder)]
        return cutted_graph
            
    def bais_calculation(self,padded_target,padded_tample,align_range):#输入pad过的目标和模板，返回x和y的偏移量。
        
        target_fft = np.fft.fft2(padded_target)
        tample_fft = np.fft.fft2(padded_tample)
        conv2 = np.real(np.fft.ifft2(target_fft*tample_fft))
        conv_height,conv_width = np.shape(conv2)
        y_center,x_center = (int((conv_height-1)/2),int((conv_width-1)/2))
        find_location = conv2[(y_center-align_range):(y_center+align_range),(x_center-align_range):(x_center+align_range)]
        y_bais = np.where(find_location ==np.max(find_location))[0][0] -align_range# 得到偏移的y量。
        x_bais = np.where(find_location ==np.max(find_location))[1][0] -align_range# 得到偏移的x量。
        ##这里的返回值，y+意味着需要向下移动图；x+意味着需要向右移动图。
        return[x_bais,y_bais]
        
    def main(self,target,tample,tample_boulder = 20,align_range = 20):
        
        target_boulder = int(tample_boulder+np.floor(align_range*1.5))
        cutted_target = self.boulder_cut(target,target_boulder)
        target_height,target_width = np.shape(cutted_target)
        cutted_tample = self.boulder_cut(tample,tample_boulder)
        tample_height,tample_width = np.shape(cutted_tample)
        target_pad = np.pad(np.rot90(cutted_target,2),((0,tample_height-1),(0,tample_width-1)),'constant')
        tample_pad = np.pad(cutted_tample,((0,target_height-1),(0,target_width-1)),'constant')#减一的目的是把矩阵奇数话，这样一定会有一个中心点。
        [x_bais,y_bais] = self.bais_calculation(target_pad,tample_pad,align_range)
        temp_baised_graph = np.pad(target,((align_range+y_bais,align_range-y_bais),(align_range+x_bais,align_range-x_bais)),'median')
        baised_graph = temp_baised_graph[align_range:-align_range,align_range:-align_range]#这个是移动后的图
        return baised_graph
        
    
    
    
    
    
    


#%% Test functions below.        
if __name__ == '__main__':

    #test = Graph_Processing.Graph_File_Average(global_tif_name[20:50])
    print('Test Run Ended.\n')
