# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:55:31 2019

@author: ZR
这个程序是一个变种，用于plotbar，可以生成framedata和celldata，并作t图。
"""
import functions_OD as pp
import cv2
import numpy as np

bar_folder = save_folder+r'\Bar_Data'
pp.mkdir(bar_folder)
#%%首先作frame的t图。
frame_set = np.zeros()



#%%
cell_condition_data = np.zeros(shape = (np.shape(spike_train)[0],len(stim_set)),dtype = np.float64)#得到每个细胞的condition tuning 数据
for i in range(0,np.shape(spike_train)[0]):#循环细胞
    for j in range(0,len(stim_set)):#循环全部condition
        temp_frame = Frame_Stim_Check[str(stim_set[j])]#当前condition的全部帧id
        cell_condition_data[i,j] = spike_train[i,temp_frame[:]].mean()
    
