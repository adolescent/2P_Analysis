# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:29:22 2019

@author: adolescent
"""
import numpy as np
import cv2
import scipy.ndimage
import functions_OD as pp
import dill
import skimage.morphology
import skimage.measure
import time

dill.load_session('Step1_Variable.pkl')#载入前一个任务的变量，方便变量继承
#%%
class Cell_Found():#定义类
    name =r'Cell_Found'#定义class的属性，如果没有__init__的内容就会以这里的作为类属性。
    def __init__(self,show_gain,save_folder,thres):
        self.save_folder = save_folder#保存目录
        self.show_gain = show_gain#show gain
        self.thres = thres#分细胞阈值
        
    def Gauss_generation(self):#生成一些初始变量
        model_frame = cv2.imread((self.save_folder+'\\Graph_After_Align.tif'),-1)#以对齐后平均作为标准
        self.model = np.float64(model_frame)/self.show_gain#这是平均图
        self.H1 = pp.normalized_gauss2D([7,7],1.5)
        self.H2 = pp.normalized_gauss2D([15,15],7)
        self.bouder = np.mean(self.model)#以平均亮度作为边界的填充值
        
    def graph_binary(self):#将图像二值化处理，并得到细胞信息图
        #切一个20像素的边框
        self.model[0:20,:] = self.bouder
        self.model[492:512,:] = self.bouder
        self.model[:,0:20] = self.bouder
        self.model[:,492:512] = self.bouder
        #高斯模糊背景和细胞，得到分细胞根据
        im_sharp = scipy.ndimage.correlate(self.model,self.H1,mode = 'nearest')
        im_back = scipy.ndimage.correlate(self.model,self.H2,mode = 'nearest')
        im_cell = (im_sharp-im_back)/np.max(im_sharp-im_back)
        level = np.mean(im_cell)+self.thres*np.std(im_cell)
        ret,bw = cv2.threshold(im_cell,level,255,cv2.THRESH_BINARY)
        bw = np.bool_(bw)#将图像二值化
        bw_clear = skimage.morphology.remove_small_objects(bw,20,connectivity = 1)#去掉面积小于20个像素的区域
        cell_label = skimage.measure.label(bw_clear)# 找到不同的连通区域，并用数字标注区域编号。
        self.cell_group = skimage.measure.regionprops(cell_label)# 将这些细胞分别得出来。
        #关于cell_group操作的注释：a[i].coords:得到连通区域i的坐标,y,x；a[i].convex_area:得到连通区域的面积；a[i].centroid:得到连通区域的中心坐标y,x
    def cell_wash(self):#清洗细胞，对X或Y大于35的就过滤掉
        for i in range(len(self.cell_group)-1,-1,-1):#注意pop掉之后全部id会移动
            temp_y = self.cell_group[i].coords[:,0]
            temp_x = self.cell_group[i].coords[:,1]
            y_range = temp_y.max()-temp_y.min()
            x_range = temp_x.max()-temp_x.min()
            if (y_range>35 or x_range>35):
                self.cell_group.pop(i)
    def show_cell(self):#由于前面筛掉了一些cell，所以这里图需要重新画。
        thres_graph = np.zeros(shape = (512,512),dtype = np.uint8)
        for i in range(len(self.cell_group)):
            x_list,y_list = pp.cell_location(self.cell_group[i])
            thres_graph[y_list,x_list] = 255
        RGB_graph = cv2.cvtColor(thres_graph,cv2.COLOR_GRAY2BGR)#转灰度为RGB
        base_graph_path = self.save_folder+'\\cell_graph'
        cv2.imwrite(base_graph_path+'.tif',RGB_graph)
        cv2.imwrite(base_graph_path+'resized.tif',cv2.resize(RGB_graph,(1024,1024))) #把细胞图放大一倍并保存起来
       # pp.show_cell(base_graph_path+'.tif',self.cell_group)# 在细胞图上标上细胞的编号。
        pp.show_cell(base_graph_path+'resized.tif',self.cell_group)# 在细胞图上标上细胞的编号。
    def main(self):#主函数，一次完成执行工作。
        self.Gauss_generation()
        self.graph_binary()
        self.cell_wash()
        self.show_cell()
        
if __name__ == '__main__':
    start_time = time.time()#任务开始时间
    show_gain = 32
    save_folder = r'D:\datatemp\190412_L74\test_data\results'
    cf = Cell_Found(show_gain,save_folder,1)#这两个变量可以从上一步里读出来
    cf.main()
    cell_group = cf.cell_group
    variable_name = 'Step2_Variable.pkl'
    dill.dump_session(variable_name)
    finish_time = time.time()
    print('Task Time Cost:'+str(finish_time-start_time)+'s')
    