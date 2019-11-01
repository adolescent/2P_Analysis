# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:56:49 2019

@author: ZR
专门为四倍镜准备，做拼图用的小程序。alpha版，只返回输入目录下的平均图，不对齐。

"""

import General_Functions.OS_Tools as OS_Tools
import General_Functions.Graph_Tools as Graph_Tools

class Puzzles(object):
    
    name = r'4x puzzles'
    
    def __init__(self,path):
        
        self.all_tif_names = OS_Tools.Path_Control.file_name(path,file_type = '.tif')
        self.save_folder = data_folder+r'\save_folder'
        OS_Tools.Path_Control.mkdir(self.save_folder)
        
    def graph_plot(self):
        
        temp_Graph = Graph_Tools.Graph_Processing.Graph_File_Average(self.all_tif_names,Formation = 'f8')
        temp_Graph = Graph_Tools.Graph_Processing.Graph_Clip(temp_Graph,2.5,Formation = 'f8')
        self.Averaged_Graph = Graph_Tools.Graph_Processing.Graph_Normalization(temp_Graph,bit = 'u2')
        Graph_Tools.Graph_Processing.Write_Graph(self.save_folder,self.Averaged_Graph,'Averaged_Graph',graph_formation = '.png',wait_time = 5000)
        
        
        
        
        
if __name__ == '__main__':
    
    data_folder = r'E:\ZR\Data_Temp\191026_L69_LM\1-009'
    P = Puzzles(data_folder)
    P.graph_plot()
    test = P.Averaged_Graph
    