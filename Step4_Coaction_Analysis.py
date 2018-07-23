# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:04:22 2018
@author: ZR
This part is the analysis of cell coaction.
Algorithm:
    1.Determine the cell is spiking or not(二值化发放数据)2std
    2.对全序列进行聚类分析，并画出每个cluster的平均发放。
    3对每个cluster出现的位置进行记录并频率分析，得到每个cluster出现的频率图。
使用的变量
save_folder
cell_group
"""
#%% 读入spike_train
import pickle
import function_in_2p as pp
import numpy as np
fr = open(save_folder+'\\spike_train','rb')
spike_train = pickle.load(fr)
cell_Num,Frame_Num = np.shape(spike_train)
unwanted_cell_index = [0,1,2,22,45,105,181,285,146,357,358]
#%% 将 spike_Train二值化 
binary_spike_train = np.zeros(shape = (cell_Num,Frame_Num))
for cell_id in range(0,cell_Num):
    temp_spike = spike_train[cell_id,:]
    temp_average = np.mean(temp_spike)
    temp_std = np.std(temp_spike)
    single_unit_spike_train = np.bool_(temp_spike*(temp_spike>(temp_average+2*temp_std)))
    binary_spike_train[cell_id,:] =single_unit_spike_train 
#%% 去除不想要的cell序列，将这些序列的二值化spike_train全都置为零。
zero_list = np.zeros(shape = Frame_Num)
for i in range(0,len(unwanted_cell_index)):
    binary_spike_train[unwanted_cell_index[i],:] = zero_list
#%% 尝试聚类，分出来合适的patterm
binary_spike_train = np.transpose(binary_spike_train) #每个横行是一个case，纵列是不同case
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as clus_h
Z = clus_h.linkage(binary_spike_train,method = 'ward',optimal_ordering= bool)#聚类运算 
fw = open((save_folder+'\\Clustered_Data'),'wb')
pickle.dump(Z,fw)#保存细胞连通性质的变量。
#%%之后对Z进行运算，包括画出各种树形图
#第一幅是无修的柱状图。根据这一幅图的结果调整接下来的分类过程。
plt.figure(figsize = (25,10))
plt.title('Hierarchincal Clustering Dendrogram(Full)')
plt.ylabel('distance')
clus_h.dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.savefig(save_folder+r'\\Dendrogram(Full).png')
plt.show()
#%% 第一幅图完成之后，根据实际情况选择第一个distance，然后标注距离阈值，节点和横坐标个数
plt.figure(figsize = (25,10))
pp.fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=20, #plot最后多少个节点
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,  # 最小标注的距离
    #max_d = 10 #水平截止线
)
plt.savefig(save_folder+r'\\Dendrogram(Last 20 Nodes).png')
plt.show()
#%% 确定截止位置，画出截止线。
plt.figure(figsize = (25,10))
pp.fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=16, #plot最后多少个节点
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,  # 最小标注的距离
    max_d = 11.5 #水平截止线
)
plt.savefig(save_folder+r'\\Dendrogram(Distance Determination).png')
plt.show()
#%%确定聚类分类,并得到每个cluster的发放pattern。
clusters = clusters = clus_h.fcluster(Z, 11.5, criterion='distance')
fw = open((save_folder+'\\Clusters'),'wb')
pickle.dump(clusters,fw)#保存细胞连通性质的变量。
averaged_graph = np.zeros(shape = (np.max(clusters),cell_Num))#从这里开始注意加1
for i in range(0,np.max(clusters)):
    frame_id = np.where(clusters ==(i+1))[0]#cluster是从1开始的
    cluster_temp =binary_spike_train[frame_id,:]
    averaged_graph[i,:] = np.mean(cluster_temp,axis = 0)