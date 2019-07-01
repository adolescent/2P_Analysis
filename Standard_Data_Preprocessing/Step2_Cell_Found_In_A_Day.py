# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:53:07 2019

@author: ZR
"""
import numpy as np
import cv2
import scipy.ndimage
import General_Functions.my_tools as pp
import skimage.morphology
import skimage.measure
import time

class Cell_Find_A_Day():
    
    def __init__(self,show_gain,root_data_folder,run_lists,thres,model_frame_name,save_name):#不写并行，似乎不能提高计算速度？
        self.thres = thres
        self.data_folder = []
        self.save_folder = []
        self.show_gain = show_gain
        self.model_frame_name = model_frame_name
        self.save_name = save_name
        for i in range(len(run_lists)):#把每一个datafolder拼接在一起
            self.data_folder.append(root_data_folder+'\\1-'+run_lists[i]) #这里是数据子文件夹的结构
            self.save_folder.append(self.data_folder[i]+r'\\results')
            
    def Gauss_generation(self):#生成一些初始变量
        model_frame = cv2.imread((self.save_folder[0]+'\\'+self.model_frame_name),-1)
        #model = pp.read_variable(self.save_folder[0]+'\\Global_Average_graph.pkl')#都一样，就选第一个了
        self.model = np.float64(model_frame)/self.show_gain#pkl文件没有增益，不用除
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
            if (y_range>25 or x_range>25):#大于25的就认为不是细胞了
                self.cell_group.pop(i)
    def cell_plot(self):
        thres_graph = np.zeros(shape = (512,512),dtype = np.uint8)
        for i in range(len(self.cell_group)):
            x_list,y_list = pp.cell_location(self.cell_group[i])
            thres_graph[y_list,x_list] = 255
        RGB_graph = cv2.cvtColor(thres_graph,cv2.COLOR_GRAY2BGR)#转灰度为RGB
        for i in range(len(self.save_folder)):
            cv2.imwrite(self.save_folder[i]+r'\\'+self.save_name+'.tif',RGB_graph)
            cv2.imwrite(self.save_folder[i]+r'\\'+self.save_name+'_resized.tif',cv2.resize(RGB_graph,(1024,1024)))
            pp.show_cell(self.save_folder[i]+r'\\'+self.save_name+'_resized.tif',self.cell_group)# 在细胞图上标上细胞的编号。
            pp.save_variable(self.cell_group,self.save_folder[i]+r'\\'+self.save_name+'.pkl')
            
if __name__ == '__main__':
    start_time = time.time()
    show_gain = 32
    root_data_folder = r'E:\ZR\Data_Temp\190514_L74_LM'
    run_lists = ['002']
    model_frame_name = 'After_Align_Global.tif'
    save_name = 'Cell_Graph_Morphology'
    thres = 1.5
    CFA = Cell_Find_A_Day(show_gain,root_data_folder,run_lists,thres,model_frame_name,save_name)
    CFA.Gauss_generation()
    CFA.graph_binary()
    CFA.cell_wash()
    CFA.cell_plot()
    finish_time = time.time()
    print('Time cost:'+str(finish_time-start_time))
    