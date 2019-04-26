# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:13:00 2019

@author: ZR
画出来pattern
"""

import pickle
import functions_cluster as pp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

def read_variable(name):#读取变量用的题头，希望这个可以在后续删掉
    with open(name, 'rb') as file:
        variable = pickle.load(file)
    file.close()
    return variable


class Pattern_Plot():
    
    name =r'Plot the pattern map'
    
    def __init__(self,save_folder,averaged_patterns,cell_group,Frame_Cluster_Information):#初始化，输入变量
        self.averaged_patterns = averaged_patterns
        self.cell_group = cell_group
        self.Frame_Cluster_Information = Frame_Cluster_Information
        self.pattern_folder = save_folder+r'\\Patterns'
        pp.mkdir(self.pattern_folder)
        self.pattern_Num,self.cell_Num = np.shape(self.averaged_patterns)
        
    def pattern_normalization(self):#归一化，在这里选择对全部都的pattern自己的最大最小值归一化。
        self.normalized_patterns = np.zeros(shape = (self.pattern_Num,self.cell_Num),dtype = np.float64)
        for i in range(0,self.pattern_Num):#对每一个pattern分别归一化
            temp_max = self.averaged_patterns[i,:].max()
            temp_min = self.averaged_patterns[i,:].min()
            self.normalized_patterns[i,:] = (self.averaged_patterns[i,:]-temp_min)/(temp_max-temp_min)
            
    def pattern_threshold(self,thres):#对归一化之后的pattern进行阈值处理，突出体现其特征。
        self.thres_pattern = self.normalized_patterns
        for i in range(0,self.pattern_Num):
            temp_list = self.normalized_patterns[i,:]
            temp_mean = temp_list.mean()
            temp_std = temp_list.std()
            temp_thres = temp_mean+thres*temp_std
            self.thres_pattern[self.thres_pattern<temp_thres]=0
            
            
    def pattern_reduction(self):#将pattern还原为图片
        for i in range(0,self.pattern_Num):#循环各个patterm
            current_graph = np.zeros(shape = (512,512))
            weight = self.thres_pattern[i,:]
            for j in range(0,len(weight)):#循环一个patterm的各个细胞
                x_list,y_list = pp.cell_location(self.cell_group[j])
                current_graph[y_list,x_list] = weight[j]*255
            pattern_Name = self.pattern_folder+r'\Pattern'+str(i+1)
            #接下来绘图
            current_graph_labled = cv2.cvtColor(np.uint8(current_graph),cv2.COLOR_GRAY2BGR)
            cv2.imwrite(pattern_Name+'.png',np.uint8(cv2.resize(current_graph_labled,(1024,1024))))
            pp.show_cell(pattern_Name+'.png',self.cell_group)
    def frame_count(self):#这个函数用来数每个pattern中的帧数目，用于参考pattern代表性。
        unique, counts = np.unique(self.Frame_Cluster_Information, return_counts=True)
        f = open(self.pattern_folder+'\\Frame_Count.txt','w')
        for i in range(0,len(counts)):
            f.write('Pattern ID:'+str(i+1)+', Frame Count:'+str(counts[i])+'\n')
        f.close()
    def bar_prop_plot(self):#绘制一个简单的箱型图，用来简要描述pattern出现的时间信息。
        pattern_separation = np.zeros(shape = (self.pattern_Num,len(self.Frame_Cluster_Information)))
        for i in range(0,len(self.Frame_Cluster_Information)):
             pattern_id = self.Frame_Cluster_Information[i]-1
             pattern_separation[pattern_id,i] = 1
        #之后对每个横行画图，分两个subplot，得到箱图。
        axprops = dict(xticks=[], yticks=[])
        barprops = dict(aspect='auto', cmap=plt.cm.binary, interpolation='nearest')
        fig = plt.figure(figsize = (80,10))
        # a horizontal barcode
        for i in range(0,self.pattern_Num):
            ax = fig.add_axes([0,1-(i+1)*0.98/self.pattern_Num,0.99,1/(self.pattern_Num+1)], **axprops)#最后两位是bar的长度和高度，前两位是这个bar在图片中心点的坐标 
            ax.imshow(pattern_separation[i].reshape((1,-1)), **barprops)
        plt.savefig(save_folder+'\Pattern_Trains.png')
        #plt.show()
            
        
if __name__ == '__main__':
    start_time = time.time()
    save_folder = read_variable('save_folder.pkl')#读入保存目录
    averaged_patterns = read_variable('averaged_patterns.pkl')#读入pattern样式
    cell_group = read_variable('cell_group.pkl')#读入细胞位置文件
    Frame_Cluster_Information = read_variable('Frame_Cluster_Information.pkl')#读入每帧归属的pattern文件
    
    pa = Pattern_Plot(save_folder,averaged_patterns,cell_group,Frame_Cluster_Information)
    pa.pattern_normalization()#归一化
    pa.pattern_threshold(0.5)#得到阈值后的pattern
    pa.pattern_reduction()#还原成图片
    #%%
    pa.frame_count()
    pa.bar_prop_plot()
    
    
    
    finish_time = time.time()
    print('Task Time Cost:'+str(finish_time-start_time)+'s')
    
    
    