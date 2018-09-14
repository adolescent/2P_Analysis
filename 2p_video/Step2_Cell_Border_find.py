# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 10:34:23 2018

@author: LvLab_ZR


使用的变量
save_folder
show_gain
"""
#%% 第一节，将图像二值化，方便后续操作,基准是对齐之后的平均图。
import numpy as np
import cv2
import scipy.ndimage
import function_in_2p as pp
import skimage.morphology
import skimage.measure
#%% Initializing
show_gain = show_gain #由于是重新读取了前面的图，所以show_gain要和之前保持一致
thren = 1 # 分细胞的阈值，高过分布多少个标准差的认为是细胞？
save_folder = save_folder#这一行最后要改
model_frame = cv2.imread((save_folder+'\\Graph_Afrer_Align.tif'),-1)#-1模式保留原始16位信息
model = np.float64(model_frame/show_gain)
H1 = pp.normalized_gauss2D([7,7],1.5)
H2 = pp.normalized_gauss2D([15,15],7)
bouder = np.mean(model)
#%% 图像二值化处理
#切一个20像素的边框
model[0:20,:] = bouder
model[492:512,:] = bouder
model[:,0:20] = bouder
model[:,492:512] = bouder
#高斯模糊背景和细胞，得到分细胞根据
im_sharp = scipy.ndimage.correlate(model,H1,mode = 'nearest')
im_back = scipy.ndimage.correlate(model,H2,mode = 'nearest')
im_cell = (im_sharp-im_back)/np.max(im_sharp-im_back)
level = np.mean(im_cell)+thren*np.std(im_cell)
ret,bw = cv2.threshold(im_cell,level,255,cv2.THRESH_BINARY)
bw = np.bool_(bw)#将图像二值化
bw_clear = skimage.morphology.remove_small_objects(bw,20,connectivity = 1)#去掉面积小于20个像素的区域
#%%之后对上图进行边界识别。

