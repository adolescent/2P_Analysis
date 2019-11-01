# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:07:16 2019

@author: ZR
"""

import cv2
import numpy as np
import General_Functions.my_tools as pp
from scipy import stats
import random
import scipy.ndimage as scimg

class Graph_Generation():

    name = 'Generate functional map'
    
    def __init__(self,stim_set_A,stim_set_B,map_name,save_folder,cell_find_type,spike_train,cell_group):
        
        self.map_folder = save_folder+r'\\Stim_Graphs_'+cell_find_type
        pp.mkdir(self.map_folder)
        self.stim_set_A = stim_set_A
        self.stim_set_B = stim_set_B
        self.map_name = map_name
        self.spike_train = pp.read_variable(spike_train)
        self.cell_group = pp.read_variable(cell_group)
    
    def ID_Configuration(self):
        
        Frame_Stim_Check = pp.read_variable(save_folder+r'\\Frame_Stim_Check.pkl')
        self.frame_set_A = []
        self.frame_set_B = []
        for i in range(len(self.stim_set_A)):
            self.frame_set_A.extend(Frame_Stim_Check[self.stim_set_A[i]])
        for i in range(len(self.stim_set_B)):
            self.frame_set_B.extend(Frame_Stim_Check[self.stim_set_B[i]])
            
    def Sub_Map(self):
        
        aligned_tif_name = pp.read_variable(save_folder+r'\\aligned_tif_name.pkl')
        average_frame_A = np.zeros(shape = (512,512),dtype = np.float64)
        average_frame_B = np.zeros(shape = (512,512),dtype = np.float64)
        for i in range(0,len(self.frame_set_A)):#得到A的平均图
            temp_frame = np.float64(cv2.imread(aligned_tif_name[self.frame_set_A[i]],-1))
            average_frame_A = average_frame_A + temp_frame/len(self.frame_set_A)
        for i in range(0,len(self.frame_set_B)):#得到B的平均图
            temp_frame = np.float64(cv2.imread(aligned_tif_name[self.frame_set_B[i]],-1))
            average_frame_B = average_frame_B + temp_frame/len(self.frame_set_B)
        #接下来做减图和clip
        sub_graph = average_frame_A - average_frame_B
        clip_min = sub_graph.mean()-3*sub_graph.std()
        clip_max = sub_graph.mean()+3*sub_graph.std()
        sub_graph_clipped = np.clip(sub_graph,clip_min,clip_max)#对减图进行最大和最小值的clip
        norm_sub_graph = (sub_graph_clipped-sub_graph_clipped.min())/(sub_graph_clipped.max()-sub_graph_clipped.min())
        #保存原始图片
        pp.save_variable(norm_sub_graph,self.map_folder+r'\\'+self.map_name+'_Graph.pkl')
        #以上得到了clip且归一化了的map
        real_sub_map = np.uint8(np.clip(norm_sub_graph*255,0,255))
        pp.save_graph(map_name,real_sub_map,self.map_folder,'.png',8,1)
        #接下来画滤波后的
        sub_map_filtered = scimg.filters.gaussian_filter(real_sub_map,1)
        pp.save_graph(map_name+'_Filtered',sub_map_filtered,self.map_folder,'.png',8,1)

        
    def Cell_Graph(self):
        
# =============================================================================
#       定义在一开始进行         
#         self.spike_train = pp.read_variable(save_folder+r'\\spike_train.pkl')
#         self.cell_group = pp.read_variable(save_folder+r'\\Cell_Group.pkl')
# =============================================================================
        cell_tuning = np.zeros(shape = (np.shape(self.spike_train)[0],1),dtype = np.float64)
        for i in range(np.shape(self.spike_train)[0]):#全部细胞循环
            temp_cell_A = 0
            for j in range(len(self.frame_set_A)):#叠加平均刺激setA下的细胞反应
                temp_cell_A = temp_cell_A+self.spike_train[i,self.frame_set_A[j]]/len(self.frame_set_A)
            temp_cell_B = 0
            for j in range(len(self.frame_set_B)):
                temp_cell_B = temp_cell_B+self.spike_train[i,self.frame_set_B[j]]/len(self.frame_set_B)
            cell_tuning[i] = temp_cell_A-temp_cell_B
        pp.save_variable(cell_tuning,self.map_folder+r'\\'+self.map_name+'_Cells.pkl')#把这个刺激的cell data存下来
        #至此得到的cell tuning是有正负的，我们按照绝对值最大把它放到-1~1的范围里
        norm_cell_tuning = cell_tuning/abs(cell_tuning).max()
        clip_min_cell = norm_cell_tuning.mean()-3*norm_cell_tuning.std()
        clip_max_cell = norm_cell_tuning.mean()+3*norm_cell_tuning.std()
        cell_clipped = np.clip(norm_cell_tuning,clip_min_cell,clip_max_cell)#对减图进行最大和最小值的clip
        #接下来plot出来细胞减图
        sub_graph_cell = np.ones(shape = (512,512),dtype = np.float64)*127
        for i in range(len(cell_clipped)):
            x_list,y_list = pp.cell_location(self.cell_group[i])
            sub_graph_cell[y_list,x_list] = (cell_clipped[i]+1)*127
        pp.save_graph(self.map_name+'_Cell',sub_graph_cell,self.map_folder,'.png',8,1)
        
    def Tuning_Index(self):
        
        preference_index = np.zeros(shape = (np.shape(self.spike_train)[0],1),dtype = np.float64)
        for i in range(0,np.shape(self.spike_train)[0]):#全部细胞循环
            temp_cell_A = 0
            for j in range(0,len(self.frame_set_A)):#叠加平均刺激setA下的细胞反应
                temp_cell_A = temp_cell_A+self.spike_train[i,self.frame_set_A[j]]/len(self.frame_set_A)
            temp_cell_B = 0
            for j in range(0,len(self.frame_set_B)):
                temp_cell_B = temp_cell_B+self.spike_train[i,self.frame_set_B[j]]/len(self.frame_set_B)
            preference_index[i] = temp_cell_A-temp_cell_B
            norm_preference_index = preference_index/abs(preference_index).max()
        index_graph = np.zeros(shape = (512,512,3),dtype = np.uint8)
        for i in range(0,len(preference_index)):
            if preference_index[i]>0:
                x_list,y_list = pp.cell_location(self.cell_group[i])
                index_graph[y_list,x_list,2] = norm_preference_index[i]*255#注意CV2读写颜色顺序是BGR= =
                #index_graph[y_list,x_list,2] = norm_preference_index[i]*255
            else:
                x_list,y_list = pp.cell_location(self.cell_group[i])
                index_graph[y_list,x_list,0] = abs(norm_preference_index[i])*255
        #绘图
        pp.save_graph(self.map_name+r'_Tuning_Index',index_graph,self.map_folder,'.png',8,1)
        
    def T_Test_Map(self):
        t_graph = np.zeros(shape = (512,512,3),dtype = np.uint8)
        cell_t = []#定义每个细胞的AB之差t值。效应量= t/sqrt(N)t值越大A越强，为负则越小B越强。
        cell_p = []#定义每个细胞AB差异的显著性p
        cell_effect_size = []#定义每个细胞的AB效应量。
        for i in range(0,np.shape(self.spike_train)[0]):#全部细胞循环
            set_size = min(len(self.frame_set_A),len(self.frame_set_B))#先定义配对的样本大小
            temp_A_set = random.sample(list(self.spike_train[i,self.frame_set_A[:]]),set_size)#在两个刺激下，都随机选择N个
            temp_B_set = random.sample(list(self.spike_train[i,self.frame_set_B[:]]),set_size)
            [temp_t,temp_p] = stats.ttest_rel(temp_A_set,temp_B_set)
            cell_t.append(temp_t)
            cell_p.append(temp_p)#按顺序把p和t加进来
            if temp_p<0.05:#显著检验
                    cell_effect_size.append(temp_t/np.sqrt(set_size))
            else:
                    cell_effect_size.append(0)
        norm_cell_effect_size = cell_effect_size/abs(np.asarray(cell_effect_size)).max()
        for i in range(0,len(norm_cell_effect_size)):
            if norm_cell_effect_size[i]>0:#大于零为红，小于零为蓝
                x_list,y_list = pp.cell_location(self.cell_group[i])
                t_graph[y_list,x_list,2] = norm_cell_effect_size[i]*255#注意CV2读写颜色顺序是BGR= =
                #index_graph[y_list,x_list,2] = norm_preference_index[i]*255
            else:
                x_list,y_list = pp.cell_location(self.cell_group[i])
                t_graph[y_list,x_list,0] = abs(norm_cell_effect_size[i])*255
        pp.save_graph(self.map_name+r'_T_Graph',t_graph,self.map_folder,'.png',8,1)
        
        
        
if __name__ =='__main__':
    
    save_folder = r'E:\ZR\Data_Temp\191026_L69_LM\1-010\results'        
    set_A = ['3','7']#这里画图画的是A-B
    set_B = ['0']
    map_name = 'Orien0-0'        
    cell_find_type = 'Morphology'
    spike_train_name = 'spike_train_'+cell_find_type+'.pkl'
    cell_group_name = 'Cell_Groups_'+cell_find_type+'.pkl'
    spike_train = save_folder+r'\\'+spike_train_name
    cell_group = save_folder+r'\\'+cell_group_name
    GG = Graph_Generation(set_A,set_B,map_name,save_folder,cell_find_type,spike_train,cell_group)
    GG.ID_Configuration()
    GG.Sub_Map()
    GG.Cell_Graph()
    GG.Tuning_Index()
    GG.T_Test_Map()
