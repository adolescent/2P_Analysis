# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:40:50 2019

@author: ZR

"""
import numpy as np


#%%
import cv2
import General_Functions.Graph_Tools as Graph_Tools
test = Graph_Tools.Graph_Processing.Graph_Normalization(clipped_t_matrix,bit = 'u2')

cv2.imshow('test',test)
cv2.waitKey(7500)
cv2.destroyAllWindows()
#%%
import General_Functions.OS_Tools as OS_Tools
T_Test_Result = OS_Tools.Save_And_Read.read_variable(r'E:\ZR\Data_Temp\191106_L69_OI\Run14_G8\Results\T_Test_Result.pkl')

#%%
import General_Functions.Graph_Tools as Graph_Tools
p_matrix = current_T_result['p_value']
t_matrix = current_T_result['t_value']
sig_matrix = (p_matrix < sig_thres) #显著的pix为True，不显著的为False,这个相当于一个显著性mask，可以把不显著的结果都mask掉了
clip_t_matrix = Graph_Tools.Graph_Processing.Graph_Clip(t_matrix,2.5)
sig_t_matrix = clip_t_matrix*sig_matrix #这里只保留了显著的部分。
sig_t_graph = Graph_Tools.Graph_Processing.Graph_Colorization(sig_t_matrix,show_time = 7500)

#%%
import matplotlib.pyplot as plt
t_matrix = T_Test_Result_OD['O-A']['t_value']
clip_t_matrix = Graph_Tools.Graph_Processing.Graph_Clip(t_matrix,2.5)
all_t = clip_t_matrix.reshape(-1,)
plt.hist(all_t,bins = 50)
#%%
import General_Functions.OS_Tools as OS_Tools
spike_train = OS_Tools.Save_And_Read.read_variable(r'E:\ZR\Data_Temp\191026_L69_LM\1-010\results\spike_train_Morphology.pkl')