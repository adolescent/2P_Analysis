# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:42:59 2019

@author: ZR

This Function will do correlation between different stimulus maps, generate a correlation matrix to 
explain cross-correlation between different ways.

From now this program can only work on cell data with type of pkl.
"""
import pandas as pd
import numpy as np
import General_Functions.my_tools as pp
import matplotlib.pyplot as plt
import seaborn
#import cv2

class Correlation_Matrix(object):
    
    name = 'Correlation between input variables'
    
    def __init__(self,variable_name,graph_folder):
        
        self.variable_name = variable_name
        self.all_variable = {} 
        self.graph_folder = graph_folder
        
    def add_element(self,current_path):     
        
           temp_graph = list(pp.read_variable(current_path)[:,0])#把数据一维化,一定要注意格式
           temp_name = current_path.split('\\')[-1].split('.')[0][:-6]
           self.all_variable[temp_name] = temp_graph
           
    def data_generation(self):
        
        for i in range(len(all_graph_name)):
            self.add_element(all_graph_name[i])
        data_frame = pd.DataFrame(self.all_variable,dtype = np.float64)
        self.correlation_matrix = data_frame.corr()
        
    def correlation_plot(self):
        plt.figure(figsize = (20,15))
        seaborn.set(font_scale = 2)
        seaborn.heatmap(self.correlation_matrix, center=0, annot=True)
        plt.savefig(graph_folder+r'\\Correlation_Map.png')
        plt.show()
            
    
    #%%
if __name__ == '__main__':
    graph_folder = r'E:\ZR\Data_Temp\190412_L74_LM\All-Stim-Maps\Run02'
    all_graph_name = pp.file_name(graph_folder,'.pkl')
    CM = Correlation_Matrix(all_graph_name,graph_folder)
    CM.data_generation()
    CM.correlation_plot()
# =============================================================================
#     a = pd.DataFrame(test_data,dtype=np.float64)
#    test_data = CM.all_variable
#     #%%
#     a_corr = a.corr()
# =============================================================================
    #%%

    seaborn.heatmap(a_corr, center=0, annot=True)
    mp.show()