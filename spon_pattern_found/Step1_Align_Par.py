# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 12:39:30 2019

@author: ZR
一个工程化的代码方式，利用定义类等模式进行了这一步操作。
self开头的变量是类变量，可以被函数外调用
"""
import functions_cluster as pp
import cv2
import numpy as np
import time
import multiprocessing as mp
from functools import partial
import pickle

class Align_Tifs:#定义类，即align
    name =r'Align_All_Frames'#定义class的属性，如果没有__init__的内容就会以这里的作为类属性。
    
    def __init__(self,data_folder,show_gain,core_Num):#初始化变量
        self.data_folder = data_folder#文件目录
        self.show_gain = show_gain#对齐增益，GA取32，RG取256
        self.core_Num = core_Num#用于多核运算的核心数，在这里指定。
        
    def path_cycle(self):#这里的变量从self自己那里来,需要returnall_tif_name和tif_Num
        self.all_tif_name = pp.tif_name(self.data_folder)
        self.save_folder = self.data_folder+r'\results'
        pp.mkdir(self.save_folder)
        self.aligned_frame_folder = self.save_folder+'\Aligned_Frames' #保存对齐过后图片的文件夹
        pp.mkdir(self.aligned_frame_folder)
        self.frame_Num = len(self.all_tif_name)#计数，总tif数
        print('There are ',self.frame_Num,'tifs in total.\n')
        
    def frame_average(self,tif_name):#把一个目录下的全部tif绘制平均图。注意要输入全部tifname的那个
        averaged_frame = np.empty(shape = [512,512])
        for i in range(20,len(tif_name)):
            averaged_frame_count = len(tif_name)-20#去掉前20帧
            temp_frame = cv2.imread(tif_name[i],-1)
            averaged_frame += (temp_frame/averaged_frame_count)
        return averaged_frame#返回值是平均之后的帧。 
        
    def show_graph(self,graph_name,graph):#这个函数用于画图并保存在目录里
        cv2.imshow(graph_name,np.uint16(np.clip(np.float64(graph)*self.show_gain,0,65535)))#加了clip
        cv2.waitKey(2500)
        cv2.destroyAllWindows()
        cv2.imwrite((self.save_folder+r'\\'+graph_name+'.tif'),np.uint16(np.clip(np.float64(graph)*self.show_gain,0,65535)))
        
    def align(self,i,base_frame):#对齐主程序。需要输入基准帧，一般以对齐前平均为基准
       # for i in range(0,len(self.all_tif_name)): #并行掉了
        temp_tif = cv2.imread(self.all_tif_name[i],-1)
        [x_bias,y_bias] = pp.bias_correlation(temp_tif,base_frame)
        temp_biased_graph = np.pad(temp_tif,((20+y_bias,20-y_bias),(20+x_bias,20-x_bias)),'constant')
        biased_graph = temp_biased_graph[20:532,20:532]
        cv2.imwrite((self.aligned_frame_folder+'\\'+self.all_tif_name[i].split('\\')[-1]),biased_graph)
    #接下来三个函数用于避免pickle/dill的时候报错
    
    def pool_set(self):#将进程池设为成员变量，这样就可以在dill的时候删除了
        self.pool = mp.Pool(self.core_Num)
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        if 'pool' in self_dict:
            del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
    
    def save_variable(self,variable,name):
        fw = open(name,'wb')
        pickle.dump(variable,fw)#保存细胞连通性质的变量。 
        fw.close()
        
    def main(self):
        self.path_cycle()
        graph_before_align = self.frame_average(self.all_tif_name)
        self.show_graph('Graph_Before_Align',graph_before_align)
        print('Start Align...')
        partial_align = partial(self.align,base_frame = graph_before_align)
        self.pool_set()
        self.pool.map(partial_align,range(len(self.all_tif_name)))
        self.pool.close()
        self.pool.join()
        print('Frame_Align_Done!')
        self.aligned_frame_name = pp.tif_name(self.aligned_frame_folder)#定义类变量，方便提取
        self.graph_after_align = self.frame_average(self.aligned_frame_name)
        self.show_graph('Graph_After_Align',self.graph_after_align)
    
#%%运行部分
if __name__ == '__main__':
    start_time = time.time()#任务开始时间
    show_gain = 32#图片增益
    data_folder = r'G:\ZR\data_processing\190412_L74_LM\1-001'
    run = Align_Tifs(data_folder,show_gain,5)
    run.main()
    graph_after_align = run.graph_after_align
    save_folder = run.save_folder
    aligned_frame_name = run.aligned_frame_name
    #%%接下来将变量保存下来,保存在代码所在目录
    run.save_variable(aligned_frame_name,'aligned_frame_name.pkl')
    run.save_variable(show_gain,'show_gain.pkl')
    run.save_variable(save_folder,'save_folder.pkl')  
    run.save_variable(graph_after_align,'graph_after_align.pkl') 
    finish_time = time.time()
    print('Task Time Cost:'+str(finish_time-start_time)+'s')
