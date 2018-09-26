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
import gc#利用gc.collect()可回收内存。
save_folder = save_folder
show_gain = show_gain
loaded_graph = pickle.load(open(save_folder+r'\\Aligned_All_Graph','rb'))#把全部帧载入。
all_graph = loaded_graph[19:493,19:493,15000:20000]#把边去掉，同时取15000-20000为实验部分。
boulder = cv2.imread(save_folder+r'\\boulder.tif',-1)
del loaded_graph
gc.collect()
# 以下是输入参数。
bins = 1 #几张一bin，1即不作bin
frame_rate = 8 #几帧一秒，注意这个是按照bin过之后计算的，5bin5frame_rate就是25原始帧一秒。
sub_flag = 1 #1为减掉每像素本底的，0为不减本底的
boulder_flag = 0 #1为标注细胞边界，0则不进行标注
#%% 第一个部分对data进行取bin处理。调试状态为了节省时间，就不删除all_graph了
print('Start Bining....')
if bins !=1:
    frame_Num = np.shape(all_graph)[2]
    binned_frame_Num = frame_Num//bins
    binned_all_graph = np.zeros(shape = (474,474,binned_frame_Num))
    for i in range(0,binned_frame_Num):
        binned_all_graph[:,:,i] = np.uint16(np.mean(all_graph[:,:,i*bins:(i+1)*bins],axis = 2))
    del all_graph
    gc.collect()
else:
    binned_all_graph = all_graph
    del all_graph
    gc.collect()
#%% 第二部分是确定做原图还是做减图.这一步输出应该是归一化增益过的八位图
print('Binned,Start subing..')
sub_all_graph = np.zeros(shape = (474,474,np.shape(binned_all_graph)[2]),dtype = np.uint8)
if sub_flag == 1:
    averaged_frame = np.mean(binned_all_graph,axis = 2)
    sub_matrix = np.zeros(shape = (474,474,np.shape(binned_all_graph)[2]),dtype = np.float64)#这个是全体dF/F组成的矩阵
    #减图办法利用逐图片归一化方法，即每张图根据它的最大和最小进行。
    for i in range(0,474):#这个循环得到了减图全部元素组成的矩阵
        for j in range(0,474):
            pix_mean = np.mean(binned_all_graph[i,j,:])
            pix_std = np.std(binned_all_graph[i,j,:])
            temp_pix = np.clip(binned_all_graph[i,j,:],0,pix_mean+3*pix_std)#取三个标准差的clip
            pix_mean = np.mean(temp_pix)#clip 之后重新计算平均值。
            sub_matrix[i,j,:] = (temp_pix-pix_mean)/pix_mean
#下一步是对以上减图进行逐步归一化处理。
    for i in range(0,np.shape(binned_all_graph)[2]):
        sub_all_graph[:,:,i] = np.uint8(pp.normalize_vector(sub_matrix[:,:,i])*255)
    del binned_all_graph       
    gc.collect()     
else:#如果flag不做，那就是直接将图片转为八位即可。
    for i in range(0,474):#为了节约计算资源，防止内存占用过度，这里使用较耗时的for循环方法。
        for j in range(0,474):
            temp_pix = np.float64(binned_all_graph[i,j,:])    
            sub_all_graph[i,j,:] = np.uint8(np.clip(temp_pix*show_gain/256,0,255))
    del binned_all_graph
    gc.collect()
#到这一步结束，输出sub_all_graph，是已经有增益了的八位图。
#%% 第三部分是在上一步基础上标注出来细胞的存在
print('Sub Complete,Start Marking...')
marked_all_graph = np.zeros(shape = (474,474,np.shape(sub_all_graph)[2]),dtype = np.uint8)
if boulder_flag ==0:
    marked_all_graph = sub_all_graph
    del sub_all_graph
    gc.collect()
else:
    for i in range(0,np.shape(sub_all_graph)[2]):
        marked_all_graph[:,:,i] = np.uint8(np.clip(np.uint16(sub_all_graph[:,:,i])+np.uint16(boulder/2),0,255))
    del sub_all_graph
    gc.collect()
#%% 第四部分，最后把上面的变量按照约定的帧数写入视频。
print('Marked,Start Video Writing....')
video_name = save_folder+r'\\Sub_Video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MP42')
videoWriter = cv2.VideoWriter(video_name,fourcc,frame_rate,(474,474),0)
for i in range(0,np.shape(marked_all_graph)[2]):
    img = marked_all_graph[:,:,i]
    videoWriter.write(img)
print('Finished,Video File at'+video_name)
del videoWriter