# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 10:34:23 2018

@author: ZR
"""
#%% 第一节，将图像二值化，方便后续操作,基准是对齐之后的平均图。
import numpy as np
import cv2
import scipy.ndimage
import function_in_2p as pp
show_gain = 32 #由于是重新读取了前面的图，所以show_gain要和之前保持一致
model_frame = cv2.imread(r'D:\datatemp\180508_L14\Run02_spon\1-002\save_folder_for_py\Graph_Afrer_Align.tif',-1)
model = np.float64(model_frame/show_gain)
H1 = pp.normalized_gauss2D([7,7],1.5)
H2 = pp.normalized_gauss2D([15,15],7)
bouder = np.mean(model)
#切一个20像素的边框
model[0:20,:] = bouder
model[492:512,:] = bouder
model[:,0:20] = bouder
model[:,492:512] = bouder
im_sharp = scipy.ndimage.correlate(model,H1,mode = 'nearest')
im_back = scipy.ndimage.correlate(model,H1,mode = 'nearest')
