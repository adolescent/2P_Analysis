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
import functions_video as pp
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
Spike_prop = np.float64(np.sum(sub_matrix>Spike_thres,axis=2)/np.shape(sub_matrix)[2])#每个像素发放帧所占的比例
#Spike_prop = np.float64(np.sum(sub_matrix>1,axis = 2)/np.shape(sub_matrix)[2])
On_Off_Graph = np.uint16(np.clip(pp.normalize_vector(Spike_prop)*65535,0,65535))
cv2.imshow('On-Off Data',On_Off_Graph)
cv2.waitKey(5000)#图片出现的毫秒数
cv2.destroyAllWindows()
cv2.imwrite(save_folder+r'\\On-Off_Graph_0.5std.png',On_Off_Graph)#这个图就是用On-Off得到的相应地图。
#%%这里对以上图片进行找细胞的操作，具体和Step1的找细胞方法类似。
Cell_thres = 2#分细胞阈值，以一个标准差以上作为细胞的依据。
H1 = pp.normalized_gauss2D([7,7],1.5)
H2 = pp.normalized_gauss2D([15,15],7)
im_sharp = np.float64(scipy.ndimage.correlate(On_Off_Graph,H1,mode = 'nearest'))
im_back =  np.float64(scipy.ndimage.correlate(On_Off_Graph,H2,mode = 'nearest'))#化成float64以免相减的时候负数溢出变得巨亮
im_cell = np.clip((im_sharp-im_back)/np.max(im_sharp-im_back),0,1)
level = np.mean(im_cell)+Cell_thres*np.std(im_cell)
ret,bw = cv2.threshold(im_cell,level,255,cv2.THRESH_BINARY)
bw = np.bool_(bw)#将图像二值化
bw_clear = skimage.morphology.remove_small_objects(bw,20,connectivity = 1)#去掉面积小于20个像素的区域
blank = np.zeros(shape = (512,512),dtype = bool)
blank[19:493,19:493] = bw_clear
bw_clear = blank
#%%找到连通区域，即细胞的位置。
cell_label = skimage.measure.label(bw_clear)# 找到不同的连通区域，并用数字标注区域编号。
cell_group = skimage.measure.regionprops(cell_label)# 将这些细胞分别得出来。
#关于cell_group操作的注释：a[i].coords:得到连通区域i的坐标,y,x；a[i].convex_area:得到连通区域的面积；a[i].centroid:得到连通区域的中心坐标y,x
import pickle
fw = open((save_folder+'\\On_Off_cell_group'),'wb')
pickle.dump(cell_group,fw)#保存细胞连通性质的变量。 
#%%绘图，并把细胞的编号画出来。
thres_graph = np.uint8(bw_clear)*255 #二值化图像的表示化为RGB
RGB_graph = cv2.cvtColor(thres_graph,cv2.COLOR_GRAY2BGR)
base_graph_path = save_folder+'\\On_Off_cell_graph.tif'
cv2.imwrite(base_graph_path,cv2.resize(RGB_graph,(1024,1024))) #把细胞图放大一倍并保存起来
pp.show_cell(base_graph_path,cell_group)# 在细胞图上标上细胞的编号。
cv2.imwrite(save_folder+r'\On_Off_cell_Origin.png',thres_graph)