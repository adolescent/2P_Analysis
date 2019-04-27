# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:24:05 2019

@author: ZR
不只需要图。。。应该考虑把graphdata和细胞data也存出来
"""

#%% 在这里输入刺激ID,我们可以做出dF/F的减图。
import cv2
import numpy as np
import functions_OD as pp

stim_set_A = ['4','8']
stim_set_B = ['0']
graph_name = 'Orien135-0'
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
OD_map = np.uint8(np.clip(norm_sub_graph*255,0,255))
cv2.imshow('Sub_Frame',OD_map)
cv2.imwrite(save_folder+r'\\'+graph_name+'.png',OD_map)
cv2.waitKey(2500)
cv2.destroyAllWindows()
#%%对以上的OD减图做双边滤波
OD_map_filtered = cv2.bilateralFilter(OD_map,9,41,41)
cv2.imshow('Sub_Frame_Filtered',OD_map_filtered)
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
OD_map_cell = np.uint8(norm_sub_graph_cell*256)
cv2.imshow('Cell_Graph',OD_map_cell)
cv2.imwrite(save_folder+'\\'+graph_name+'_cell.png',OD_map_cell)
cv2.waitKey(2500)
cv2.destroyAllWindows()
#接下来对细胞图做tuning的伪彩色图。
OD_map_cell_color=cv2.applyColorMap(cv2.convertScaleAbs(OD_map_cell),cv2.COLORMAP_HOT)
cv2.imshow('Cell_Graph_Color',OD_map_cell_color)
cv2.imwrite(save_folder+'\\'+graph_name+'_cell_color.png',OD_map_cell_color)
cv2.waitKey(2500)
cv2.destroyAllWindows()
#%% 接下来考虑做一个tuningIndex的图。
preference_index = np.zeros(shape = (np.shape(spike_train)[0],1),dtype = np.float64)
#定义tuningIndex = (A-B)/(A+B),为正即Atuning，为负是Btuning。
for i in range(0,np.shape(spike_train)[0]):#全部细胞循环
    temp_cell_A = 0
    for j in range(0,len(frame_set_A)):#叠加平均刺激setA下的细胞反应
        temp_cell_A = temp_cell_A+spike_train[i,frame_set_A[j]]/len(frame_set_A)
    temp_cell_B = 0
    for j in range(0,len(frame_set_B)):
        temp_cell_B = temp_cell_B+spike_train[i,frame_set_B[j]]/len(frame_set_B)
    preference_index[i] = temp_cell_A-temp_cell_B
#接下来对两个轴归一化，把两个方向最大的一个归为1.
    norm_preference_index = preference_index/abs(preference_index).max()
#得到index的序列，接下来上色。
    #%%
index_graph = np.zeros(shape = (512,512,3),dtype = np.uint8)
for i in range(0,len(preference_index)):
    if preference_index[i]>0:
        x_list,y_list = pp.cell_location(cell_group[i])
        index_graph[y_list,x_list,2] = norm_preference_index[i]*255#注意CV2读写颜色顺序是BGR= =
        #index_graph[y_list,x_list,2] = norm_preference_index[i]*255
    else:
        x_list,y_list = pp.cell_location(cell_group[i])
        index_graph[y_list,x_list,0] = abs(norm_preference_index[i])*255
#绘图
cv2.imshow('Cell_Tuning_Graph',index_graph)
cv2.imwrite(save_folder+'\\'+graph_name+'_cell_tuning_index.png',index_graph)
cv2.waitKey(2500)
cv2.destroyAllWindows() 
#%%接下来做一个ttest的图。一个配对t检验，通过了再画否则skip。
from scipy import stats
import random
t_graph = np.zeros(shape = (512,512,3),dtype = np.uint8)
cell_t = []#定义每个细胞的AB之差t值。效应量= t/sqrt(N)t值越大A越强，为负则越小B越强。
cell_p = []#定义每个细胞AB差异的显著性p
cell_effect_size = []#定义每个细胞的AB效应量。
for i in range(0,np.shape(spike_train)[0]):#全部细胞循环
    set_size = min(len(frame_set_A),len(frame_set_B))#先定义配对的样本大小
    temp_A_set = random.sample(list(spike_train[i,frame_set_A[:]]),set_size)#在两个刺激下，都随机选择N个
    temp_B_set = random.sample(list(spike_train[i,frame_set_B[:]]),set_size)
    [temp_t,temp_p] = stats.ttest_rel(temp_A_set,temp_B_set)
    cell_t.append(temp_t)
    cell_p.append(temp_p)#按顺序把p和t加进来
    if temp_p<0.05:#显著检验
            cell_effect_size.append(temp_t/np.sqrt(set_size))
    else:
            cell_effect_size.append(0)
norm_cell_effect_size = cell_effect_size/abs(np.asarray(cell_effect_size)).max()
for i in range(0,len(norm_cell_effect_size)):
    if norm_cell_effect_size[i]>0:#大于零为红，小于零为蓝
        x_list,y_list = pp.cell_location(cell_group[i])
        t_graph[y_list,x_list,2] = norm_cell_effect_size[i]*255#注意CV2读写颜色顺序是BGR= =
        #index_graph[y_list,x_list,2] = norm_preference_index[i]*255
    else:
        x_list,y_list = pp.cell_location(cell_group[i])
        t_graph[y_list,x_list,0] = abs(norm_cell_effect_size[i])*255
#%%绘图
cv2.imshow('Cell_Tuning_T-test',t_graph)
cv2.imwrite(save_folder+'\\'+graph_name+'_cell_preference_t_graph.png',t_graph)
cv2.waitKey(2500)
cv2.destroyAllWindows() 