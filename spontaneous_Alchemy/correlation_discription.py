# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:33:08 2019

@author: ZR
This tool will calculate the person correlation between a stimulus map and a set of clustered datas
"""
import General_Functions.my_tools as pp
import numpy as np
import cv2

class Correlation_Description(object):
    
    name = 'Correlation calculation'
    
    def __init__(self,target_graph,clustered_data):
        self.target_graph = target_graph
        self.clustered_data = clustered_data
        
    def calculation_unit(self,i):#计算单元，计算当前component i 和目标图的相似度。
        
        
        
if __name__ == '__main__':
    save_folder = r'E:\ZR\Data_Temp\190412_L74_LM\1-002\results'
    target_graph = pp.read_variable(r'E:\ZR\Data_Temp\190412_L74_LM\All-Stim-Maps\Run02\A-O_Cells.pkl')
    clustered_data = pp.read_variable(save_folder+r'\PCAed_Data.pkl')
    CD = Correlation_Descripion(target_graph,clustered_data)