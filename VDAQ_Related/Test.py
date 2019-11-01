# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:40:50 2019

@author: ZR

"""
import numpy as np

data = np.fromfile(r'E:\ZR\Data_Temp\180629_L63_OI_Run01_G8_Test\G8_E00B000.BLK', dtype='<u4')[429:]
#%%
all_datas = np.reshape(data,(-1,540,654),)
#%%
sample_frame = all_datas[5,:,:]
sample_frame = Graph_Tools.Graph_Processing.Graph_Clip(sample_frame,1.5)
norm_sample_fram = (sample_frame-sample_frame.min())/(sample_frame.max()-sample_frame.min())
test_graph = (norm_sample_fram*65535).astype('u2')
#%%
import cv2
cv2.imshow('test',normed_filtered_graph)
cv2.waitKey(7500)
cv2.destroyAllWindows()
#%% 
import General_Functions.my_tools as pp
import scipy.ndimage
HP = pp.normalized_gauss2D([2,2],2)
LP = pp.normalized_gauss2D([80,80],80)
HP_graph = scipy.ndimage.correlate(test_graph.astype('f8'),HP,mode = 'nearest')
LP_graph = scipy.ndimage.correlate(test_graph.astype('f8'),LP,mode = 'nearest')
filtered_graph = HP_graph-LP_graph
#%%
import General_Functions.Graph_Tools as Graph_Tools
cliped_filtered_graph = Graph_Tools.Graph_Processing.Graph_Clip(filtered_graph,1.5)
normed_filtered_graph = Graph_Tools.Graph_Processing.Graph_Normalization(cliped_filtered_graph,bit = 'u2')