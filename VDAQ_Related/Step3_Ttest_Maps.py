# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:35:57 2019

@author: ZR

"""

import numpy as np
from scipy import stats
import random
import General_Functions.OS_Tools as OS_Tools
import General_Functions.Graph_Tools as Graph_Tools
import General_Functions.Filters as Filters

class T_Test_Map(object):
    
    def __init__(self,Head_Property,Produced_data,sub_parameter,save_path):
        
        self.save_path = save_path
        self.Head_Property = Head_Property
        self.Produced_data = Produced_data
        self.sub_parameter = sub_parameter
        
    def t_data_reshape(self,data_set_A,data_set_B):#输入两组数据，返回筛选并reshape之后的数据。
        
        if np.shape(data_set_A)[0] == np.shape(data_set_B)[0]:#如果两个组一样多的话
            map_num = np.array(np.shape(data_set_A)[0])
            A_set_reshapped = data_set_A.reshape(map_num,-1,)
            B_set_reshapped = data_set_B.reshape(map_num,-1,)
            return A_set_reshapped,B_set_reshapped
        else:#如果两组data不一样多，则取最少的那一组。
            map_num = min(np.shape(data_set_A)[0],np.shape(data_set_B)[0])
            A_selection_ids = random.sample(list(range(np.shape(data_set_A)[0])),map_num)
            B_selection_ids = random.sample(list(range(np.shape(data_set_B)[0])),map_num)
            A_set_reshapped = data_set_A[A_selection_ids].reshape(map_num,-1,)
            B_set_reshapped = data_set_B[B_selection_ids].reshape(map_num,-1,)
            return A_set_reshapped,B_set_reshapped
        
    def Single_T_Generator(self,A_set_reshapped,B_set_reshapped):#用来进行逐像素t检验，并返回一个t/p矩阵作为
        
        Graph_Height = self.Head_Property['Height']
        Graph_Width = self.Head_Property['Width']
        
        pic_num,pix_num = np.shape(A_set_reshapped)
        T_Test_result = {}
        T_Test_result['Graph_Num'] = pic_num
        T_Test_result['t_value'] = np.zeros(pix_num,dtype = np.float64)
        T_Test_result['p_value'] = np.zeros(pix_num,dtype = np.float64)
        for i in range(pix_num):#逐像素做t检验
            current_pix_A = A_set_reshapped[:,i]
            current_pix_B = B_set_reshapped[:,i]#这里需要用list，接下来才可以做t检验
            if len(set(current_pix_A)) == 1 or len(set(current_pix_B)) == 1:#排除所有过曝的点，这些点会导致std = 0，导致ttest nan
                T_Test_result['t_value'][i] = 0#所有过曝的点视作没有效应量
                T_Test_result['p_value'][i] = 1
            else:
                [T_Test_result['t_value'][i],T_Test_result['p_value'][i]] = stats.ttest_rel(current_pix_A,current_pix_B)
            
        T_Test_result['t_value'] = T_Test_result['t_value'].reshape(Graph_Height,Graph_Width)
        T_Test_result['p_value'] = T_Test_result['p_value'].reshape(Graph_Height,Graph_Width)
        return T_Test_result
        
    def Graph_Filt_Clip(self,origin_graph):#输入原始图片，在这里做clip和归一化
        
        F = Filters.Filter()
        method_key = self.sub_parameter['Filter_Method']
        hp_parameter = self.sub_parameter['HP_Filter_Parameter']
        lp_parameter = self.sub_parameter['LP_Filter_Parameter']
        clip_std = self.sub_parameter['Clip_std']
        hp_graph = F.Main(origin_graph,method_key,hp_parameter)
        lp_graph = F.Main(origin_graph,method_key,lp_parameter)
        filtered_graph = hp_graph-lp_graph
        clipped_graph = Graph_Tools.Graph_Processing.Graph_Clip(filtered_graph,clip_std)
        normed_graph = Graph_Tools.Graph_Processing.Graph_Normalization(clipped_graph,bit = 'u2')
        return normed_graph
    
    def Main(self):#主函数，做每幅图的ttest，并把结果保存下来
        
        all_keys = list(self.Produced_data.keys())
        for i in range(len(self.Produced_data)):
            current_key = all_keys[i]
            current_A_set = self.Produced_data[current_key][0]
            current_B_set = self.Produced_data[current_key][1]
            reshaped_A,reshaped_B = self.t_data_reshape(current_A_set,current_B_set)
            current_T_result = self.Single_T_Generator(reshaped_A,reshaped_B)
            OS_Tools.Save_And_Read.save_variable(current_T_result,self.save_path+r'\\'+current_key+'_T_Test_Result.pkl')
            t_graph_origin = current_T_result['t_value']/np.sqrt(current_T_result['Graph_Num'])
            t_graph = self.Graph_Filt_Clip(t_graph_origin)
            Graph_Tools.Graph_Processing.Write_Graph(self.save_path,t_graph,current_key+r'_T_Graph')
            
if __name__ == '__main__':
# =============================================================================
#     
#     TTM = T_Test_Map(Head_Property,Produced_data,Sub_parameter,save_path)
# 
#     #%%
#     T_Test_result = TTM.Single_T_Generator(a,b)
#     #%%
#     import General_Functions.Graph_Tools as Graph_Tools
#     import cv2
#     graph = Graph_Tools.Graph_Processing.Graph_Clip(graph_origin,1.5)
#     norm_graph = Graph_Tools.Graph_Processing.Graph_Normalization(graph,bit = 'u2')
#     Graph_Tools.Graph_Processing.Write_Graph(r'E:\ZR\Data_Temp\191106_L69_OI\Run01_OD8\Results',norm_graph,'test')
#     #%%
#     import General_Functions.OS_Tools as OS_Tools
#     Produced_data = OS_Tools.Save_And_Read.read_variable(r'E:\ZR\Data_Temp\191106_L69_OI\Run01_OD8\Results\Produced_data.pkl')
# =============================================================================
    print('Test Run Ended')