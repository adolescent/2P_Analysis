# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:38:17 2018

@author: ZR
这一部分主要是找到几个没有细胞的区域，同时和细胞区域的spike train进行对比，从而确认我们分出来的细胞和本底发放有所不同。
使用的变量
save_folder
"""
#%% 首先是初始化和输入载入。
import numpy as np
import functions_video as pp
import pickle
import gc
import matplotlib.pyplot as plt 
import cv2
import skimage.measure
save_folder = save_folder
Frames = pickle.load(open(save_folder+r'\\Aligned_All_Graph','rb'))[:,:,15000:20000]#这里需要原图！
cell_graph = cv2.imread((save_folder+r'\\Blank_Included.png'),0)
#%% 接下来对读入图片的联通区域进行分析和标号，手动确认非细胞和细胞区的新id
cell_label = skimage.measure.label(cell_graph)# 找到不同的连通区域，并用数字标注区域编号。
cell_group = skimage.measure.regionprops(cell_label)# 将这些细胞分别得出来。
#关于cell_group操作的注释：a[i].coords:得到连通区域i的坐标,y,x；a[i].convex_area:得到连通区域的面积；a[i].centroid:得到连通区域的中心坐标y,x
RGB_graph = cv2.cvtColor(cell_graph,cv2.COLOR_GRAY2BGR)
base_graph_path = save_folder+'\\Blank_Included_graph.tif'
cv2.imwrite(base_graph_path,cv2.resize(RGB_graph,(1024,1024))) #把细胞图放大一倍并保存起来
pp.show_cell(base_graph_path,cell_group)# 在细胞图上标上细胞的编号。
#%%总体亮度变化。随着时间推进，由于不稳定性导致的整体亮度偏移：即全域的dF/F
#%%接下来手动键入细胞的id和对照非细胞区域的id
blank_id = [5,12,31,35,78,95,106,148,152]
compare_cell_id = [47,50,82,90,100,102,104,122,133]
average_frame = np.mean(Frames,axis = 2)
#首先循环得到空白区域的dF/F Train
blank_Num = len(blank_id)
spike_train_Null = np.zeros(shape=(len(blank_id),np.shape(Frames)[2]))
for i in range(0,blank_Num):#i是细胞id
    base_F = pp.sum_a_frame(average_frame,cell_group[blank_id[i]])
    for j in range(0,np.shape(Frames)[2]):
        frame_F = pp.sum_a_frame(Frames[:,:,j],cell_group[blank_id[i]])
        spike_train_Null[i,j] = (frame_F-base_F)/base_F
#第二个循环得到细胞区域的dF/F Train
compare_cell_Num = len(compare_cell_id)
spike_train_compare_cell = np.zeros(shape=(len(compare_cell_id),np.shape(Frames)[2]))
for i in range(0,compare_cell_Num):#i是细胞id
    base_F = pp.sum_a_frame(average_frame,cell_group[compare_cell_id[i]])
    for j in range(0,np.shape(Frames)[2]):
        frame_F = pp.sum_a_frame(Frames[:,:,j],cell_group[compare_cell_id[i]])
        spike_train_compare_cell[i,j] = (frame_F-base_F)/base_F        