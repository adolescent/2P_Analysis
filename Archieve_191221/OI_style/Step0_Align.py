# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:49:38 2019

@author: ZR

Just another align step, if already aligned before, this step can be ignored.

"""

import General_Functions.Graph_Tools as Graph_Tools
import General_Functions.OS_Tools as OS_Tools
import cv2
import numpy as np
import time

class Align_In_A_Day(object):
    
    name = r'对齐所有在同一个位置的帧'
    
    def __init__(self,root_data_folder,run_lists):
        
        self.run_lists = run_lists
        self.data_folder = []
        for i in range(len(run_lists)):#把每一个datafolder拼接在一起
            self.data_folder.append(root_data_folder+'\\1-'+self.run_lists[i]) #这里是数据子文件夹的结构
            
    def folder_generation(self,upper_folder):#给定特定目录，在其下级建立result和Aligned文件夹，并返回目录
        
        result_folder = upper_folder+r'\results'
        OS_Tools.Path_Control.mkdir(result_folder)
        aligned_frame_folder = result_folder+'\Aligned_Frames' #保存对齐过后图片的文件夹
        OS_Tools.Path_Control.mkdir(aligned_frame_folder)
        return result_folder,aligned_frame_folder    
    
    def path_cycle(self):#遍历目录，把来组不同run的数据都
        
        self.run_tif_name = []#没个元素都是当前run的tif name，格式为列表
        self.global_tif_name = []#这个是全局的tif，把所有的都放到了一起之后的。
        self.save_folders = []#一样，每个元素都是当前的保存目录
        self.aligned_frame_folder = []#
        for i in range(len(self.data_folder)): #全部文件夹
             self.run_tif_name.append(OS_Tools.Path_Control.file_name(self.data_folder[i],file_type = '.tif'))#这里是append，注意保留了不同folder的结构
             self.global_tif_name.extend(OS_Tools.Path_Control.file_name(self.data_folder[i],file_type = '.tif'))#这个是extend，放到一起了
             print('There are ',len(self.run_tif_name[i]),'tifs in condition',self.run_lists[i])
             temp_result,temp_frame = self.folder_generation(self.data_folder[i])
             self.save_folders.append(temp_result)
             self.aligned_frame_folder.append(temp_frame)
             
    def Before_Align_Average(self):
        #先平均整体，按照同一个标准对齐，之后再考虑如何
        print('Averaging global graph...')
        temp_graph = Graph_Tools.Graph_Processing.Graph_File_Average(self.global_tif_name[100:])#整体平均图，用作对齐标准于是去掉了前100帧
        temp_graph = Graph_Tools.Graph_Processing.Graph_Clip(temp_graph,2.5)
        self.global_average_before = Graph_Tools.Graph_Processing.Graph_Normalization(temp_graph,bit = 'u2')
        cv2.imshow('Global_Average',self.global_average_before)
        cv2.waitKey(2500)
        cv2.destroyAllWindows()
        for i in range(len(self.data_folder)):#平均各自的run
            print('Averaging condition '+self.run_lists[i]+'...')
            
            temp_name = 'Before_Align_Run'+self.run_lists[i]      
            temp_graph = Graph_Tools.Graph_Processing.Graph_File_Average(self.run_tif_name[i])
            temp_graph = Graph_Tools.Graph_Processing.Graph_Clip(temp_graph,2.5)
            temp_graph = Graph_Tools.Graph_Processing.Graph_Normalization(temp_graph,bit = 'u2')
            #接下来保存
            Graph_Tools.Graph_Processing.Write_Graph(self.save_folders[i],temp_graph,temp_name)#当前Run的平均
            Graph_Tools.Graph_Processing.Write_Graph(self.save_folders[i],self.global_average_before,'Before_Align_Global',wait_time = 0)#整体平均，每个run存一个
    
    def Align(self):
        print('Align starts...')
        tamplate_graph = self.global_average_before
        for i in range(len(self.global_average_before)):
            current_graph = cv2.imread(self.global_tif_name[i],-1)
            current_baised_graph = Graph_Tools.Alignment(current_graph,tamplate_graph,temple_boulder = 20,align_range = 20).baised_graph
            tif_address = self.global_tif_name[i].split('\\')
            tif_address.insert(-1,'results')
            tif_address.insert(-1,'Aligned_Frames')#把tif_address定义为对齐后的地址
            tif_address_writing = r'\\'.join(tif_address)                
            cv2.imwrite(tif_address_writing,current_baised_graph)
        print('Aligning Done!')
        
    def After_Align(self):
        
        print('Generating Aligned Graphs...')
        self.aligned_tif_name = []
        global_aligned_tif_name = []
        for i in range(len(self.aligned_frame_folder)):#把每个run里的tif_name存到各自文件夹里
            self.aligned_tif_name.append(OS_Tools.Path_Control.file_name(self.aligned_frame_folder[i],file_type = '.tif'))#保留文件层级
            global_aligned_tif_name.extend(self.aligned_tif_name[i])#存一个全部tif的变量
            run_average = Graph_Tools.Graph_Processing.Graph_File_Average(self.aligned_tif_name[i],Formation = 'f8')
            run_average = Graph_Tools.Graph_Processing.Graph_Clip(run_average,2.5)
            OS_Tools.Save_And_Read.save_variable(run_average,self.save_folders[i]+r'\\Run_Average_graph.pkl')
            run_average_graph = Graph_Tools.Graph_Processing.Graph_Normalization(run_average,bit = 'u2')
            Graph_Tools.Graph_Processing.Write_Graph(self.save_folders[i],run_average_graph,'Run_Average_After',graph_formation = '.png',wait_time = 2500)
        #接下来保存全局平均，也存在每个文件里
        global_average = Graph_Tools.Graph_Processing.Graph_File_Average(global_aligned_tif_name,Formation = 'f8')
        global_average = Graph_Tools.Graph_Processing.Graph_Clip(global_average,2.5,Formation = 'f8')
        global_average_graph = Graph_Tools.Graph_Processing.Graph_Normalization(global_average,bit = 'u2') 
        for i in range(len(self.save_folders)):
            Graph_Tools.Graph_Processing.Write_Graph(self.save_folders[i],global_average_graph,'Global_Average_After',graph_formation = '.png',wait_time = 0)
            OS_Tools.Save_And_Read.save_variable(global_average,self.save_folders[i]+r'\\Global_Average_graph.pkl')
            #注意这里保存的变量是没有增益的，是原始变量
    def main(self):
        
        self.path_cycle()
        self.Before_Align_Average()
        self.Align()
        self.After_Align()
   #%% Test Run
if __name__ == '__main__':
    start_time = time.time()#任务开始时间
    root_data_folder = r'E:\ZR\Data_Temp\190412_L74_LM'
    run_lists = ['001','002']
    #run_lists = ['001','002','003','004']
    AIA = Align_In_A_Day(root_data_folder,run_lists)
    AIA.main()
    #AIA.main()
    finish_time = time.time()
    print('Aligning time cost:'+str(finish_time-start_time)+'s')