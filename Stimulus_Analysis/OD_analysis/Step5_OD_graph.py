# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:24:05 2019

@author: ZR
"""

#%% 在这里输入刺激ID,我们可以做出dF/F的减图。
import cv2
stim_set_A = ['1','3','5','7']
stim_set_B = ['2','4','6','8']
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
sub_graph_clipped = np.clip(sub_graph,sub_graph.mean(),)
#%%
cv2.imshow('test',np.uint16(average_frame_A)*32)
cv2.waitKey(5000)
cv2.destroyAllWindows()
#%%第二幅图做细胞的相应减图。