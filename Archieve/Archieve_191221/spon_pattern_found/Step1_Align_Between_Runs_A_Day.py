# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:00:48 2019

@author: ZR
这一段代码用来对齐一天中的不同Run，并生成对齐后的图。
"""

import General_Functions.my_tools as pp
import cv2
import numpy as np
import time
#import multiprocessing as mp
#from functools import partial

class Align_In_A_Day():
    
    def __init__(self,root_data_folder,show_gain,run_lists):#不写并行，似乎不能提高计算速度？
        self.show_gain = show_gain
        self.data_folder = []
        self.run_lists = run_lists
        for i in range(len(self.run_lists)):#把每一个datafolder拼接在一起
            self.data_folder.append(root_data_folder+'\\1-'+self.run_lists[i]) #这里是数据子文件夹的结构
            
    def folder_generation(self,upper_folder):#给定特定目录，在其下级建立result和Aligned文件夹，并返回目录
        result_folder = upper_folder+r'\results'
        pp.mkdir(result_folder)
        aligned_frame_folder = result_folder+'\Aligned_Frames' #保存对齐过后图片的文件夹
        pp.mkdir(aligned_frame_folder)
        return result_folder,aligned_frame_folder
    
    def path_cycle(self):#遍历目录，把来组不同run的数据都
        self.run_tif_name = []
        self.global_tif_name = []#这个是全局的tif，把所有的都放到了一起之后的。
        self.save_folder = []
        self.aligned_frame_folder = []
        for i in range(len(self.data_folder)): 
             self.run_tif_name.append(pp.tif_name(self.data_folder[i]))#这里是append，注意保留了不同folder的结构
             self.global_tif_name.extend(pp.tif_name(self.data_folder[i]))#这个是extend，放到一起了
             print('There are ',len(self.run_tif_name[i]),'tifs in condition',self.run_lists[i])
             temp_result,temp_frame = self.folder_generation(self.data_folder[i])
             self.save_folder.append(temp_result)
             self.aligned_frame_folder.append(temp_frame)
             
    def frame_average(self,tif_name):#把一个目录下的全部tif绘制平均图。tif_name这里是形参！
        averaged_frame = np.empty(shape = [512,512])
        for i in range(20,len(tif_name)):
            averaged_frame_count = len(tif_name)-20#去掉前20帧
            temp_frame = cv2.imread(tif_name[i],-1)
            averaged_frame += (temp_frame/averaged_frame_count)
        return averaged_frame#返回值是平均之后的帧。
    
    def save_graph(self,graph_name,graph,folder):#这个函数用于画图并保存在目录里
        cv2.imshow(graph_name,np.uint16(np.clip(np.float64(graph)*self.show_gain,0,65535)))#加了clip
        cv2.waitKey(2500)
        cv2.destroyAllWindows()
        cv2.imwrite(folder+r'\\'+graph_name+'.tif',np.uint16(np.clip(np.float64(graph)*self.show_gain,0,65535)))
 
    
    
    def before_align(self):#对齐前的处理，用以绘制各种图
        print('Averaging global graph...')
        global_average_before = self.frame_average(self.global_tif_name)#整体平均图，对齐用这个
        for i in range(len(self.data_folder)):#先平均各自的run
            print('Averaging condition '+self.run_lists[i]+'...')
            temp_average_frame = self.frame_average(self.run_tif_name[i])
            temp_name = 'Before_Align_Run'+self.run_lists[i]
            self.save_graph(temp_name,temp_average_frame,self.save_folder[i])#当前Run的平均
            self.save_graph('Global_Average',global_average_before,self.save_folder[i])#整体平均，每个run存一个
        return global_average_before
    
    def Align(self,base_frame):#核心代码块，用于对齐。
        print('Aligning graphs...')
        for i in range(0,len(self.global_tif_name)):
            temp_tif = cv2.imread(self.global_tif_name[i],-1)
            [x_bias,y_bias] = pp.bias_correlation(temp_tif,base_frame)
            temp_biased_graph = np.pad(temp_tif,((20+y_bias,20-y_bias),(20+x_bias,20-x_bias)),'median')
            biased_graph = temp_biased_graph[20:532,20:532]#这个是移动后的图
            tif_address = self.global_tif_name[i].split('\\')
            tif_address.insert(-1,'results')
            tif_address.insert(-1,'Aligned_Frames')#把tif_address定义为对齐后的地址
            tif_address_writing = r'\\'.join(tif_address)                
            cv2.imwrite(tif_address_writing,biased_graph)
        print('Aligning Done!')
        
    def after_align(self):#这个用来在对齐后平均。
        print('Generationg averaged graphs...')
        self.aligned_tif_name = []
        global_aligned_tif_name = []
        for i in range(len(self.aligned_frame_folder)):#把每个run里的tif_name存到各自文件夹里
            self.aligned_tif_name.append(pp.tif_name(self.aligned_frame_folder[i]))
            global_aligned_tif_name.extend(self.aligned_tif_name[i]) 
            temp_aligned_tif_name = self.save_folder[i]+r'\\aligned_tif_name.pkl'
            pp.save_variable(self.aligned_tif_name[i],temp_aligned_tif_name)
            run_average = self.frame_average(self.aligned_tif_name[i])
            pp.save_variable(run_average,self.save_folder[i]+r'\\Run_Average_graph.pkl')
            self.save_graph('After_Align_Run'+self.run_lists[i],run_average,self.save_folder[i])
        #接下来保存全局平均，也存在每个文件里
        global_average_graph = self.frame_average(global_aligned_tif_name)
        for i in range(len(self.save_folder)):
            self.save_graph('After_Align_Global',global_average_graph,self.save_folder[i])
            pp.save_variable(global_average_graph,self.save_folder[i]+r'\\Global_Average_graph.pkl')
            #注意这里保存的变量是没有增益的，是原始变量
    def main(self):
        self.path_cycle()
        global_average_before = self.before_align()
        self.Align(global_average_before)
        self.after_align()
        
#%%
        
        
if __name__ == '__main__':
    start_time = time.time()#任务开始时间
    root_data_folder = r'E:\ZR\Data_Temp\190514_L74_LM'
    run_lists = ['002']
    show_gain = 32  #GA Mode
    AIA = Align_In_A_Day(root_data_folder,show_gain,run_lists)
    AIA.main()
# =============================================================================
#     AIA.path_cycle()
# # =============================================================================
# #     run_tif_name = AIA.run_tif_name
# #     save_folder = AIA.save_folder
# #     aligned_frame_folder = AIA.aligned_frame_folder
# #     global_tif_name = AIA.global_tif_name
# # =============================================================================
#     global_average_before = AIA.before_align()
#     AIA.Align(global_average_before)
#     AIA.after_align()
# =============================================================================
    finish_time = time.time()
    print('Aligning time cost:'+str(finish_time-start_time)+'s')