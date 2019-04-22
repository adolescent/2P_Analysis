# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 12:38:41 2018

@author: LvLab_ZR
"""
"This File can read in All tif graph and align them in an average way."
#%% Walk through the file folder, get all tif files need to be dealt with.
import functions_OD as pp
import cv2
import numpy as np
import time


data_folder =r'F:\2P_Data\190412_L74_LM\1-004'
all_tif_name = pp.tif_name(data_folder)#输入数据所在的路径，并保存全部的tif
save_folder = data_folder+r'\results' #这里输入保存路径，所有的计算结果都会在这个路径里保存。
pp.mkdir(save_folder)
aligned_frame_folder = save_folder+'\Aligned_Frames' #保存对齐过后图片的文件夹
show_gain = 32 #这个是保存图片时候的增益，用来提高图片亮度使可以看清。GA取32，RG取256
pp.mkdir(aligned_frame_folder)
frame_Num = len(all_tif_name)
print('There are ',frame_Num,'tifs in total.\n')

#%% Align frames and save them in a new folder.
#%% Before Align
if frame_Num >=1100:
    last_graph_count = 1100
else:
    last_graph_count = frame_Num # 只取前一千帧的平均
averaged_frame = np.empty(shape = [512,512])
for i in range(100,last_graph_count):
    averaged_frame_count = last_graph_count-100
    temp_frame = cv2.imread(all_tif_name[i],-1)
    averaged_frame += (temp_frame/averaged_frame_count)
graph_before_align = np.uint16(averaged_frame)
#%%
#Show the graph before align
cv2.imshow('Graph_Before_Align',np.uint16(np.clip(np.float64(graph_before_align)*show_gain,0,65535)))#加了clip
cv2.waitKey(2500)
cv2.destroyAllWindows()
# and save the graph into the save folder as Graph_Before_Align.tif, all 16bit depth.
cv2.imwrite((save_folder+'\Graph_Before_Align.tif'),np.uint16(np.clip(np.float64(graph_before_align)*show_gain,0,65535)))
#%% Do the Align,method same as the MATLAB way.
print('Start Align...')
aligned_tif_name = []
for i in range(0,len(all_tif_name)):
        temp_tif = cv2.imread(all_tif_name[i],-1)
        averaged_tif = graph_before_align #定义模板和当前帧
        [x_bias,y_bias] = pp.bias_correlation(temp_tif,averaged_tif)
        temp_biased_graph = np.pad(temp_tif,((20+y_bias,20-y_bias),(20+x_bias,20-x_bias)),'constant')
        biased_graph = temp_biased_graph[20:532,20:532]
        aligned_tif_name.append((aligned_frame_folder+'\\'+all_tif_name[i].split('\\')[4]))
        cv2.imwrite((aligned_frame_folder+'\\'+all_tif_name[i].split('\\')[4]),biased_graph)
#%% 产生Align之后的图像的平均    
print('Done! \nProducing Aligned Graphs')
averaged_frame = np.empty(shape = [512,512])
for i in range(0,len(all_tif_name)):
    temp_frame = cv2.imread(aligned_tif_name[i],-1)
    averaged_frame += (temp_frame/len(all_tif_name))
graph_after_align = np.uint16(np.clip(averaged_frame,0,65535))#加了clip
#%%Show the graph after align
#
cv2.imshow('Graph_After_Align',np.uint16(np.clip(np.float64(graph_after_align)*show_gain,0,65535)))
cv2.waitKey(2500)
cv2.destroyAllWindows()
cv2.imwrite((save_folder+'\Graph_After_Align.tif'),np.uint16(np.clip(np.float64(graph_after_align)*show_gain,0,65535)))