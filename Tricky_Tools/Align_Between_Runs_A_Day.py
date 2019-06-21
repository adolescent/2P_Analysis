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
import multiprocessing as mp
from functools import partial
import pickle

class Align_In_A_Day():
    
    def __init__(self,root_data_folder,show_gain,run_lists):#不写并行，似乎不能提高计算速度？
        self.show_gain = show_gain
        self.data_folder = []
        for i in range(len(run_lists)):#把每一个datafolder拼接在一起
            self.data_folder.append(root_data_folder+'\\1-'+run_lists[i]) #这里是数据子文件夹的结构
            
    def folder_generation(self,upper_folder):#给定特定目录，在其下级建立result和Aligned文件夹，并返回目录
        result_folder = upper_folder+r'\results'
        pp.mkdir(result_folder)
        aligned_frame_folder = result_folder+'\Aligned_Frames' #保存对齐过后图片的文件夹
        pp.mkdir(aligned_frame_folder)
        return result_folder,aligned_frame_folder
    
    def path_cycle(self):#遍历目录，把来组不同run的数据都
        self.all_tif_name = []
        self.save_folder = []
        self.aligned_frame_folder = []
        for i in range(len(self.data_folder)): 
             self.all_tif_name.append(pp.tif_name(self.data_folder[i]))#这里是append，注意保留了不同folder的结构
             print('There are ',len(self.all_tif_name[i]),'tifs in condition',run_lists[i])
             temp_result,temp_frame = self.folder_generation(self.data_folder[i])
             self.save_folder.append(temp_result)
             self.aligned_frame_folder.append(temp_frame)
        
            
            
        
        
        
        
#%%
        
        
if __name__ == '__main__':
    root_data_folder = r'E:\ZR\Data_Temp\190412_L74_LM'
    run_lists = ['001','002','003','004']
    show_gain = 32#GA Mode
    AIA = Align_In_A_Day(root_data_folder,show_gain,run_lists)
    AIA.path_cycle()
    a = AIA.all_tif_name
    b = AIA.save_folder
    c = AIA.aligned_frame_folder
    
    