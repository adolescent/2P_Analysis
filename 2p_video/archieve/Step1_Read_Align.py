# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 12:38:41 2018

@author: LvLab_ZR
"""
"This File can read in All tif graph and align them in an average way."
#%% Walk through the file folder, get all tif files need to be dealt with.
import functions_video as pp
import cv2
import numpy as np

all_tif_name = pp.tif_name(r'D:\ZR\DataTemp\L63_LL_OI_2P\180810_L63_2P\Run01_V4_L8910_D240_RG_Spon')#输入数据所在的路径，并保存全部的tif
save_folder = (r'D:\ZR\DataTemp\L63_LL_OI_2P\180810_L63_2P\Run01_V4_L8910_D240_RG_Spon\for_python') #这里输入保存路径，所有的计算结果都会在这个路径里保存。
aligned_frame_folder = save_folder+'\Aligned_Frames' #保存对齐过后图片的文件夹
show_gain = 256 #这个是保存图片时候的增益，用来提高图片亮度使可以看清。GA取32，RG取256
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
graph_before_align = averaged_frame
#%%
#Show the graph before align
cv2.imshow('Graph_Before_Align',np.uint16(np.clip(graph_before_align*show_gain,0,65535)))#做一个clip，防止超过65535的回到0
cv2.waitKey(10000)
cv2.destroyAllWindows()
# and save the graph into the save folder as Graph_Before_Align.tif, all 16bit depth.
cv2.imwrite((save_folder+'\Graph_Before_Align.tif'),np.uint16(np.clip(graph_before_align*show_gain,0,65535)))
#%%将所有的原始图像保存在一个向量里面，这个大向量有助于接下来画视频之类的操作。
all_graph = np.zeros(shape = (512,512,frame_Num),dtype = np.uint16)
#%% Do the Align,method same as the MATLAB way.
print('Start Align...')
import time
start_time = time.clock()#对Align进行计时，这里是开始时间 
aligned_tif_name = []
for i in range(0,len(all_tif_name)):
        temp_tif = cv2.imread(all_tif_name[i],-1)
        averaged_tif = graph_before_align #定义模板和当前帧
        [x_bias,y_bias] = pp.bias_correlation(temp_tif,averaged_tif)
        temp_biased_graph = np.pad(temp_tif,((20+y_bias,20-y_bias),(20+x_bias,20-x_bias)),'constant')#在原来的图边上接个边，方便裁剪。
        biased_graph = temp_biased_graph[20:532,20:532]#这个变量就是移动之后的对齐目标。
        aligned_tif_name.append((aligned_frame_folder+'\\'+all_tif_name[i].split('\\')[5]))
        cv2.imwrite((aligned_frame_folder+'\\'+all_tif_name[i].split('\\')[6]),biased_graph)
        all_graph[:,:,i] = biased_graph#按顺序保存
#%% 产生Align之后的图像的平均    
print('Done! \nProducing Aligned Graphs')
stop_time = time.clock()#这里是结束时间
print('Aligned Time is:%.3f'%((stop_time-start_time)/60)+' minutes')

averaged_frame = np.mean(all_graph,axis = 2)
graph_after_align = np.uint16(averaged_frame)
#%%Show the graph before align
#
cv2.imshow('Graph_After_Align',np.uint16(np.clip(np.float64(graph_after_align)*show_gain,0,65535)))
cv2.waitKey(10000)
cv2.destroyAllWindows()
cv2.imwrite((save_folder+'\Graph_Afrer_Align.tif'),np.uint16(np.clip(np.float64(graph_after_align)*show_gain,0,65535)))
#%% 把所有对齐之后图片组成的向量保存下来
import pickle
pickle.dump(all_graph,open(save_folder+r'\\Aligned_All_Graph','wb'),protocol = 4)#这里注意需要使用方法4才可以保存大于4G的文件。wb是写二进制的意思。