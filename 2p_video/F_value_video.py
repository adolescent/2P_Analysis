# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:31:42 2019

@author: ZR

This is the determination version of F value videos. This method will generate F videos with normalization

Not show gain as usual, but clip and normalize.
filter changed into Gauss
不再乘show gain了，这里直接归一化作图

"""


import numpy as np
import General_Functions.my_tools as pp
import cv2
import scipy.ndimage as scimg#高斯滤波器
#from PIL import Image, ImageDraw, ImageFont


class Video_Write(object):
    
    name = r'Write graph videos '
    
    def __init__(self,aligned_tif_folder,filt_flag,start_frame,stop_frame,clip_std,fps):
        
        self.save_folder = save_folder
        aligned_tif_folder = save_folder+r'\Aligned_Frames'
        self.filt_flag = filt_flag# whether filter is needed
        self.selected_frame_names = pp.file_name(aligned_tif_folder,'.tif')[start_frame:stop_frame]#selected_frame_name
        self.clip_std = clip_std#+- n std to do the clip
        self.frame_Num = stop_frame-start_frame
        self.fps = fps # how many frame per second
        
    def read_in(self):# read in all targeted tifs, save them in a variable.
       
        #首先读入变量
        self.frame_sets = np.zeros(shape = (self.frame_Num,512,512),dtype = np.float64)# 初始化读入帧变量
        for i in range(self.frame_Num):
            self.frame_sets[i,:,:] = np.float64(cv2.imread(self.selected_frame_names[i],-1))#定义这一变量作为待读入的数据集
        #然后做clip
        clip_range = self.frame_sets.std()*self.clip_std#上下各clip几个std
        data_mean = self.frame_sets.mean()
        self.frame_sets = np.clip(self.frame_sets,data_mean-clip_range,data_mean+clip_range)[:,20:492,20:492]
        #然后做归一化
        self.normed_frame_sets = (self.frame_sets-self.frame_sets.min())/(self.frame_sets.max()-self.frame_sets.min())
        #之后写成uint8的形式
        self.video_contents = np.uint8(self.normed_frame_sets*255)
        #之后考虑是不是要做filt，这次换成高斯滤波
        if self.filt_flag == True:
            self.video_contents = scimg.filters.gaussian_filter(self.video_contents,1)
        #这个变量可以直接写video了
        
    def video_plot(self,file_name,file_matrix,fps):#这是一个函数，给出名字，内容矩阵,plot得到视频
        
        size = (472,472)
        fps = fps
        videoWriter = cv2.VideoWriter(self.save_folder+r'\\'+file_name,-1,fps,size,0)#last variance: is color
        for i in range(self.frame_Num):
            current_img = file_matrix[i,:,:]
            cv2.putText(current_img,'Frame'+str(start_frame+i),(10,30),cv2.FONT_HERSHEY_PLAIN,2,(255),2)
            videoWriter.write(current_img)
        del videoWriter
        
    def F_video(self):
        
        file_name = 'F_value_video.avi'
        file_matrix = self.video_contents
        self.video_plot(file_name,file_matrix,self.fps)
        
        
    
        
        
    
    
if __name__ == '__main__':
    
    filt_flag = True
    save_folder = r'E:\ZR\Data_Temp\190412_L74_LM\1-002\results'
    start_frame = 0
    stop_frame = 500
    clip_std = 10#决定做几个std的clip。
    fps = 30
    VW = Video_Write(save_folder,filt_flag,start_frame,stop_frame,clip_std,fps)
    VW.read_in()
    VW.F_video()

    
    