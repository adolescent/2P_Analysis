# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:25:47 2019

@author: ZR
做完ttest之后，可以用这个来得到显著性图。例如只标注p>特定的结果、对t值上色的方法等等。
"""

import General_Functions.OS_Tools as OS_Tools
import General_Functions.Graph_Tools as Graph_Tools
import numpy as np

class Significance_Show(object):
    
    name = r'Plot Significant pixels, and colorization.'
    
    def __init__(self,T_Test_result,save_folder,sig_thres = 0.05):
        
        self.T_Test_result = T_Test_result
        self.save_folder = save_folder
        
    def Single_Graph_Significant(self,current_T_result,sig_thres = 0.05):#输入当前序列的t test结果，进行显著性检验并绘制彩图。
        
        p_matrix = current_T_result['p_value']
        t_matrix = current_T_result['t_value']
        sig_matrix = (p_matrix < sig_thres) #显著的pix为True，不显著的为False,这个相当于一个显著性mask，可以把不显著的结果都mask掉了
        clip_t_matrix = Graph_Tools.Graph_Processing.Graph_Clip(t_matrix,2.5)
        sig_t_matrix = clip_t_matrix*sig_matrix #这里只保留了显著的部分。
        sig_t_graph = Graph_Tools.Graph_Processing.Graph_Colorization(sig_t_matrix,show_time = 7500)
        return sig_t_graph
    
    def Main(self):#主程序，用来循环所有的t值结果并得到显著性图。
        
        all_keys = list(self.T_Test_result.keys())
        
        
        
if __name__ == '__main__':
    import time
    start_time = time.time()
    T_Test_result = OS_Tools.Save_And_Read.read_variable(r'E:\ZR\Data_Temp\191106_L69_OI\Run01_OD8\Results\T_Test_Result.pkl')
    