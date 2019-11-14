# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:25:10 2019

@author: ZR
这个工具用来做减图。

"""
import numpy as np
import General_Functions.Filters as Filters
import General_Functions.Graph_Tools as Graph_Tools
import General_Functions.OS_Tools as OS_Tools

class Sub_Map_Produce(object):
    
    name = r'Produce sub map from '
    
    def __init__(self,Head_Property,Produced_data,sub_parameter,save_path):
        
        self.save_path = save_path
        self.Head_Property = Head_Property
        self.Produced_data = Produced_data
        self.sub_parameter = sub_parameter
        
    def Single_Map_Subtraction(self,current_data_set):#输入一个图的id，得到一幅图的处理后减图。
        
        A_graphs = np.mean(current_data_set[0],axis = 0)
        B_graphs = np.mean(current_data_set[1],axis = 0)
        sub_map_origin = A_graphs-B_graphs
        
        return sub_map_origin
        
    def Map_Filter_Clip(self,graph,sub_parameter):#输入减法之后的图，然后做filter和clip。
        
        F = Filters.Filter()
        method_key = sub_parameter['Filter_Method']
        hp_parameter = sub_parameter['HP_Filter_Parameter']
        lp_parameter = sub_parameter['LP_Filter_Parameter']
        clip_std = sub_parameter['Clip_std']
        hp_graph = F.Main(graph,method_key,hp_parameter)
        lp_graph = F.Main(graph,method_key,lp_parameter)
        filtered_graph = hp_graph-lp_graph
        clipped_graph = Graph_Tools.Graph_Processing.Graph_Clip(filtered_graph,clip_std)
        normed_graph = Graph_Tools.Graph_Processing.Graph_Normalization(clipped_graph,bit = 'u2')
        return normed_graph
        
    def Main(self):
        
        all_keys = list(self.Produced_data.keys())
        for i in range(len(self.Produced_data)):#循环全部的图，每幅图都做一个
            current_data_set = self.Produced_data[all_keys[i]]
            sub_map_origin = self.Single_Map_Subtraction(current_data_set)
            origin_graph_name = self.save_path+'\\'+all_keys[i]+r'_Origin.pkl'
            OS_Tools.Save_And_Read.save_variable(sub_map_origin,origin_graph_name)
            final_sub_graph = self.Map_Filter_Clip(sub_map_origin,self.sub_parameter)
            Graph_Tools.Graph_Processing.Write_Graph(self.save_path,final_sub_graph,all_keys[i]+r'_Sub',)
            

if __name__ =='__main__':
    SMP = Sub_Map_Produce(Head_Property,Produced_data,Sub_parameter,save_path)
    SMP.Main()
    #%%
# =============================================================================
#     test = SMP.Single_Map_Subtraction(current_data_set)
#     test2 = SMP.Map_Filter_Clip(test,sub_parameter)
#     #%%
#     a = Graph_Tools.Graph_Processing.Graph_Clip(test,2.5)
#     a = Graph_Tools.Graph_Processing.Graph_Normalization(a,bit = 'u1')
#     #%%
#     Graph_Tools.Graph_Processing.Write_Graph(r'E:\ZR\Data_Temp\191106_L69_OI\Run01_OD8\Results',test2,'test')
# 
# =============================================================================
