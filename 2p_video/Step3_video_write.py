# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:50:44 2018
@author: ZR
这一部分用来生成视频。可根据需要选择的参数应该有：
1.做bin的数目
2.帧率，多少帧每秒
3.是否减去本底的F（注意本底的算法应该去除极端值，高斯滑窗本底）
4.是否标注细胞的边界。

使用的变量
save_folder
show_gain
"""
#%% 初始化和import
import numpy as np
import pickle
import cv2
import functions_video as pp
save_folder = save_folder
show_gain = show_gain
all_graph = pickle.load(open(save_folder+r'\\Aligned_All_Graph','rb'))#把全部帧载入。
all_graph = all_graph[19:493,19:493,:]#把边去掉
boulder = cv2.imread(save_folder+r'\\boulder.tif',-1)
# 以下是输入参数。
bins = 1 #几张一bin，1即不作bin
frame_rate = 8 #几帧一秒，注意这个是按照bin过之后计算的，5bin5frame_rate就是25原始帧一秒。
sub_flag = 0 #1为减掉每像素本底的，0为不减本底的
boulder_flag = 1 #1为标注细胞边界，0则不进行标注
#%% 第一个部分对data进行取bin处理
if bins !=1:
    frame_Num = np.shape(all_graph)[2]
    binned_frame_Num = frame_Num//bins
    binned_all_graph = np.zeros(shape = (474,474,binned_frame_Num))
    for i in range(0,binned_frame_Num):
        binned_all_graph[:,:,i] = np.uint16(np.mean(all_graph[:,:,i*bins:(i+1)*bins],axis = 2))
    del all_graph
else:
    binned_all_graph = all_graph
    del all_graph
#%% 第二部分是确定做原图还是做减图.
sub_all_graph = np.zeros(shape = (474,474,np.shape(binned_all_graph)[2]),dtype = np.uint8)
if sub_flag == 1:
    for i in range(0,474):
        for j in range(0,474):
            pix_mean = np.mean(binned_all_graph[i,j,:])
            pix_std = np.std(binned_all_graph[i,j,:])
            temp_pix = np.clip(binned_all_graph[i,j,:],0,pix_mean+3*pix_std)#取三个标准差的clip
            pix_mean = np.mean(temp_pix)#clip 之后重新计算平均值。
            sub_pix = pp.normalize_vector((temp_pix-pix_mean)/pix_mean)
            sub_all_graph[i,j,:] = np.uint8(sub_pix*255)
    del binned_all_graph            
else:#如果flag不做，那就是直接将图片转为八位即可。
    for i in range(0,474):#为了节约计算资源，防止内存占用过度，这里使用较耗时的for循环方法。
        for j in range(0,474):
            temp_pix = binned_all_graph[i,j,:]    
            sub_all_graph[i,j,:] = np.uint8(np.clip(temp_pix*show_gain/256,0,255))
    del binned_all_graph
#到这一步结束，输出sub_all_graph，是已经有增益了的八位图。
#%% 第三部分是在上一步基础上标注出来细胞的存在
marked_all_graph = np.zeros(shape = (474,474,np.shape(sub_all_graph)[2]),dtype = np.uint8)
if boulder_flag ==0:
    marked_all_graph = sub_all_graph
    del sub_all_graph
else:
    for i in range(0,np.shape(sub_all_graph)[2]):
        marked_all_graph[:,:,i] = np.clip(np.uint16(sub_all_graph[:,:,i])+boulder,0,255)
    del sub_all_graph
#%% 第四部分，最后把上面的变量按照约定的帧数写入视频。
video_name = save_folder+r'\\Sub_Video.mp4'
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter(video_name,-1,frame_rate,(474,474),0)
for i in range(0,500):
    img = marked_all_graph[:,:,i]
    videoWriter.write(img)