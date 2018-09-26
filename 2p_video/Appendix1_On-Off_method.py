# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 15:46:54 2018

@author: ZR
这一部分主要是一个新的找细胞办法，我们利用dF/F大于某个特定阈值的比例来画图，从而得到On-Off的细胞区域
利用的变量：
save_folder
"""
#%% 首先是初始化和输入载入。
import numpy as np
import function_video as pp
import pickle
import gc
import matplotlib.pyplot as plt 
import cv2
import scipy.ndimage
import skimage.morphology
import skimage.measure
save_folder = save_folder
sub_matrix = pickle.load(open(save_folder+r'\\sub_matrix','rb'))
flat_dF = sub_matrix.flatten()#把数据一维化，统计用
#%%对dF/F进行统计，同时画出直方图来。
#第一幅图是原图
dF_mean = np.mean(flat_dF)
dF_std = np.std(flat_dF)
plt.figure(figsize = (25,10))
plt.title('dF/F distribution')
plt.hist(flat_dF, bins=200, color='steelblue',density = True )
plt.annotate(('Mean = '+str(dF_mean)+' Total std = '+str(dF_std)),xy = (12.5,0), xytext=(12.5,2))
plt.savefig(save_folder+r'\\dF_distribution.png')
plt.show()
plt.close('all')
#%% 第二幅图是局部放大的图。
dF_mean = np.mean(flat_dF)
dF_std = np.std(flat_dF)
plt.figure(figsize = (25,10))
plt.ylim(0,0.05)
plt.title('dF/F distribution,Enlarged')
plt.hist(flat_dF, bins=200, color='steelblue',density = True )
plt.savefig(save_folder+r'\\dF_distribution_Scale_enlarged.png')
plt.show()
plt.close('all')
del flat_dF
gc.collect()#释放内存
#%%到这里选定发放与否的阈值，统计每个像素的发放概率。
Spike_thres = dF_mean+0.5*dF_std
Spike_prop = np.float64(np.sum(sub_matrix>Spike_thres,axis=2)/np.shape(sub_matrix)[2])
#Spike_prop = np.float64(np.sum(sub_matrix>1,axis = 2)/np.shape(sub_matrix)[2])
On_Off_Graph = np.uint8(np.clip(pp.normalize_vector(Spike_prop)*255,0,255))
cv2.imshow('On-Off Data',On_Off_Graph)
cv2.waitKey(5000)#图片出现的毫秒数
cv2.destroyAllWindows()
cv2.imwrite(save_folder+r'\\On-Off_Graph.png',On_Off_Graph)#这个图就是用On-Off得到的相应地图。
#%%这里对以上图片进行找细胞的操作，具体和Step1的找细胞方法类似。
