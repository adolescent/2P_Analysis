# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:15:05 2019

@author: ZR
这个小工具用来计算两幅阈值图的重叠部分。只能用于阈值图，因为会预先做一个二值化的处理。
"""

import General_Functions.Graph_Tools as Graph_Tools
import cv2
#%% 输入目录和两幅图名
graph_folder = r'E:\ZR\Data_Temp\190412_L74_LM\0412-On-Off_Maps'
graph_a_name = 'Intersection.png'
graph_b_name = '2.tif'
#%% 然后得到二值化的两幅图
graph_a = cv2.imread(graph_folder+r'\\'+graph_a_name,0)>0
graph_b = cv2.imread(graph_folder+r'\\'+graph_b_name,0)>0
#%% 计算二者重叠的部分与重叠比例
intersection_part = (graph_a*graph_b)#这个是两幅图重叠的部分
intersection_graph = (intersection_part*255).astype('u1')
Graph_Tools.Graph_Processing.Write_Graph(graph_folder,intersection_graph,'Intersection_Graph_24')
inter_propotion = intersection_part.sum()/graph_b.sum()
print('Intersection Propotion = '+str(inter_propotion))