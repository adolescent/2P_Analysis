# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 13:17:38 2018

@author: ZR

这一步分两个部分：
1、把之前得到的各个pattern平均图还原到图片上并画出来
2、对每个pattern的发放时间进行描绘，并进行频域分析，试图找出这些模式发放的频率

使用的变量
save_folder
averaged_graph(上一步平均得到的pattern集合)

"""
#%% Initializing
import pickle
import function_in_2p as pp
import numpy as np
import cv2
import matplotlib.pyplot as plt
save_folder = save_folder
averaged_graph = averaged_graph
#capture_freq = 0.7 #采样频率，0.7Hz/s
fr = open(save_folder+'\\cell_group','rb')
cell_group = pickle.load(fr)
fr = open(save_folder+'\\Clusters','rb')
clusters = pickle.load(fr)
pattern_folder = save_folder+r'\\Patterns'
pp.mkdir(pattern_folder)
pattern_Num,Cell_Num = np.shape(averaged_graph)
#%% 第一步，逐个还原发放的pattern。注意这里对最亮的一个细胞做了归一化。
for i in range(0,len(averaged_graph)):#循环各个patterm
    current_graph = np.zeros(shape = (512,512))
    weight = averaged_graph[i,:]
    for j in range(0,len(weight)):#循环一个patterm的各个细胞
        x_list,y_list = pp.cell_location(cell_group[j])
        current_graph[y_list,x_list] = weight[j]
    pattern_Name = pattern_folder+'\Pattern'+str(i+1)
    #cv2.imwrite(pattern_Name+'.png',np.uint8(current_graph*255))
    #图片放大一倍，标注cell_Number
    current_graph_labled = cv2.cvtColor(np.uint8(current_graph/np.max(averaged_graph)*255),cv2.COLOR_GRAY2BGR)
    cv2.imwrite(pattern_Name+'.png',np.uint8(cv2.resize(current_graph_labled,(1024,1024))))
    pp.show_cell(pattern_Name+'.png',cell_group)# 在细胞图上标上细胞的编号。
#%% 第二步，Count Clusters 数出来各个cluster的帧数（这个基本上是之前的图的统计）
unique, counts = np.unique(clusters, return_counts=True)
f = open(pattern_folder+'\\Frame_Count.txt','w')
for i in range(0,len(counts)):
    f.write('Pattern ID:'+str(i+1)+', Frame Count:'+str(counts[i])+'\n')
f.close()
#%% 第三步,将各个patterm的发放分到不同的序列里，每个横行代表一个模式的响应。
pattern_separation = np.zeros(shape = (np.max(clusters),len(clusters)))
for i in range(0,len(clusters)):
     pattern_id = clusters[i]-1
     pattern_separation[pattern_id,i] = 1
#之后对每个横行画图，分两个subplot，得到热谱图。
axprops = dict(xticks=[], yticks=[])
barprops = dict(aspect='auto', cmap=plt.cm.binary, interpolation='nearest')
fig = plt.figure(figsize = (25,20))
# a horizontal barcode
for i in range(0,pattern_Num):
    ax = fig.add_axes([0,1-(i+1)*0.98/pattern_Num,0.99,1/(pattern_Num+1)], **axprops)#最后两位是bar的长度和高度，前两位是这个bar在图片中心点的坐标 
    ax.imshow(pattern_separation[i].reshape((1,-1)), **barprops)
plt.savefig(save_folder+'\Pattern_Trains.png')
plt.show()