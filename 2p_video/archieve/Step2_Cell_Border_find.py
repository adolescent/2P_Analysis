# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 10:34:23 2018

@author: LvLab_ZR


使用的变量
save_folder
show_gain

这部分和之前的找细胞一样，加上了边界识别的功能。
"""
#%% 第一节，将图像二值化，方便后续操作,基准是对齐之后的平均图。
import numpy as np
import cv2
import scipy.ndimage
import functions_video as pp
import skimage.morphology
import skimage.measure
import pickle
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
#%%找到连通区域，即细胞的位置。
cell_label = skimage.measure.label(bw_clear)# 找到不同的连通区域，并用数字标注区域编号。
cell_group = skimage.measure.regionprops(cell_label)# 将这些细胞分别得出来。
#关于cell_group操作的注释：a[i].coords:得到连通区域i的坐标,y,x；a[i].convex_area:得到连通区域的面积；a[i].centroid:得到连通区域的中心坐标y,x
fw = open((save_folder+'\\cell_group'),'wb')
pickle.dump(cell_group,fw)#保存细胞连通性质的变量。 
#%%绘图，并把细胞的编号画出来。
thres_graph = np.uint8(bw_clear)*255 #二值化图像的表示化为RGB
RGB_graph = cv2.cvtColor(thres_graph,cv2.COLOR_GRAY2BGR)
extended_RGB_graph = cv2.resize(RGB_graph,(1024,1024))
base_graph_path = save_folder+'\\cell_graph.tif'
cv2.imwrite(base_graph_path,extended_RGB_graph) #把细胞图保存起来，不做标记。
pp.show_cell(base_graph_path,cell_group)#另存为细胞图，并坐上标记。
#%% 之后根据cell_graph 将此图片的边界识别出来。边界识别使用canny方法。
boulder_base = thres_graph[19:493,19:493]	
boulder = cv2.Canny(boulder_base,10,150)
cv2.imshow('boulder', boulder)
cv2.waitKey(5000)#图片出现的毫秒数
cv2.destroyAllWindows()
cv2.imwrite(save_folder+r'\\boulder.tif',boulder)