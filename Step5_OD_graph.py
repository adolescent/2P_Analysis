# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:24:05 2019

@author: ZR
"""

#%% 在这里输入刺激ID,我们可以做出dF/F的减图。
import cv2
import numpy as np
stim_set_A = ['3','4']
stim_set_B = ['7','8']
graph_name = 'OA'
frame_set_A = []#把两个刺激态里的frameid整理出来
frame_set_B = []
for i in range(0,len(stim_set_A)):
    frame_set_A.extend(Frame_Stim_Check[stim_set_A[i]])
for i in range(0,len(stim_set_B)):
    frame_set_B.extend(Frame_Stim_Check[stim_set_B[i]])
    
#%%第一幅图可以做简单的全成像范围，方法类似成像的。
#首先读入两个set的图片文件并平均。
average_frame_A = np.zeros(shape = (512,512),dtype = np.float64)
for i in range(0,len(frame_set_A)):
        temp_frame = cv2.imread(aligned_tif_name[frame_set_A[i]],-1)
        average_frame_A = average_frame_A + temp_frame
average_frame_A = average_frame_A/len(frame_set_A)
average_frame_B = np.zeros(shape = (512,512),dtype = np.float64)
for i in range(0,len(frame_set_B)):
        temp_frame = cv2.imread(aligned_tif_name[frame_set_B[i]],-1)
        average_frame_B = average_frame_B + temp_frame
average_frame_B = average_frame_B/len(frame_set_B)
#做减图之后clip再归一化
sub_graph = average_frame_A - average_frame_B
clip_min = sub_graph.mean()-3*sub_graph.std()
clip_max = sub_graph.mean()+3*sub_graph.std()
sub_graph_clipped = np.clip(sub_graph,clip_min,clip_max)#对减图进行最大和最小值的clip
norm_sub_graph = (sub_graph_clipped-sub_graph_clipped.min())/(sub_graph_clipped.max()-sub_graph_clipped.min())
#%%保存OD图
OD_map = np.uint8(norm_sub_graph*256)
cv2.imshow('test',OD_map)
cv2.imwrite(save_folder+r'\\'+graph_name+'.png',OD_map)
cv2.waitKey(2500)
cv2.destroyAllWindows()
#%%对以上的OD减图做双边滤波
OD_map_filtered = cv2.bilateralFilter(OD_map,9,41,41)
cv2.imshow('test',OD_map_filtered)
cv2.imwrite(save_folder+'\\'+graph_name+'_filtered.png',OD_map_filtered)
cv2.waitKey(2500)
cv2.destroyAllWindows()
#%%第二幅图做细胞的相应减图。
cell_tuning = np.zeros(shape = (np.shape(spike_train)[0],1),dtype = np.float64)
for i in range(0,np.shape(spike_train)[0]):#全部细胞循环
    temp_cell_A = 0
    for j in range(0,len(frame_set_A)):#叠加平均刺激setA下的细胞反应
        temp_cell_A = temp_cell_A+spike_train[i,frame_set_A[j]]/len(frame_set_A)
    temp_cell_B = 0
    for j in range(0,len(frame_set_B)):
        temp_cell_B = temp_cell_B+spike_train[i,frame_set_B[j]]/len(frame_set_B)
    cell_tuning[i] = temp_cell_A-temp_cell_B
norm_cell_tuning = (cell_tuning-cell_tuning.min())/(cell_tuning.max()-cell_tuning.min())#将细胞响应归一化
#%%接下来plot出来减图。

sub_graph_cell = np.zeros(shape = (512,512),dtype = np.float64)
for i in range(0,len(norm_cell_tuning)):
    x_list,y_list = pp.cell_location(cell_group[i])
    sub_graph_cell[y_list,x_list] = norm_cell_tuning[i]
clip_min_cell = sub_graph_cell.mean()-3*sub_graph.std()
clip_max_cell = sub_graph_cell.mean()+3*sub_graph.std()
sub_graph_cell_clipped = np.clip(sub_graph_cell,clip_min_cell,clip_max_cell)#对减图进行最大和最小值的clip
norm_sub_graph_cell = (sub_graph_cell_clipped-sub_graph_cell_clipped.min())/(sub_graph_cell_clipped.max()-sub_graph_cell_clipped.min())
#%%
OD_map_cell = np.uint8(norm_sub_graph_cell*256)
cv2.imshow('test',OD_map_cell)
cv2.imwrite(save_folder+'\\'+graph_name+'_cell.png',OD_map_cell)
cv2.waitKey(2500)
cv2.destroyAllWindows()
#%%接下来对细胞图做tuning的伪彩色图。
OD_map_cell_color=cv2.applyColorMap(cv2.convertScaleAbs(OD_map_cell),cv2.COLORMAP_HOT)
cv2.imshow('test',OD_map_cell_color)
cv2.imwrite(save_folder+'\\'+graph_name+'_cell_color.png',OD_map_cell_color)
cv2.waitKey(2500)
cv2.destroyAllWindows()