# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:39:36 2019

@author: ZR
Video Writer

Use the key video type to generate specific video
####################################################################################
'F': F value video, show what we saw online
'dF': dF video, show variance of graphs
'dF_F':Show On-Off video

"""

import numpy as np
import General_Functions.my_tools as pp
import cv2
#from PIL import Image, ImageDraw, ImageFont


class Video_Write(object):
    
    name = r'Write F video, in order to reshow online maps'
    
    def __init__(self,aligned_tif_folder,start_frame,stop_frame,show_gain,filt):
        
        self.filt = filt
        self.show_gain = show_gain
        self.all_tif_name = pp.file_name(aligned_tif_folder,'.tif')
        self.selected_frames = self.all_tif_name[start_frame:stop_frame]
        self.frame_Num = stop_frame-start_frame
        
    def read_in(self):
        
        self.frame_sets = np.zeros(shape = (self.frame_Num,512,512),dtype = np.float64)
        for i in range(self.frame_Num):
            self.frame_sets[i,:,:] = cv2.imread(self.selected_frames[i],-1)*self.show_gain/256#定义这一变量作为待读入的数据集
        self.frame_sets = np.uint8(self.frame_sets[:,20:492,20:492])
        self.average_graph = self.frame_sets.mean(axis = 0)
        #Till now, we get the aim frames #self.frame_sets#
        
    def video_plot(self,file_name,file_matrix):#Input filename,file data, generate target video in return.
        
        img_root = aligned_tif_folder.split('\\')[:-1]
        save_folder = '\\'.join(img_root)
        fps = 4
        size = (472,472)
        videoWriter = cv2.VideoWriter(save_folder+r'\\'+file_name,-1,fps,size,0)#last variance: is color
        for i in range(self.frame_Num):
            current_img = file_matrix[i,:,:]
            if self.filt == True:
                current_img = cv2.bilateralFilter(current_img,9,41,41)
            ##########Test Context below   
            cv2.putText(current_img,'Frame'+str(start_frame+i),(10,30),cv2.FONT_HERSHEY_PLAIN,2,(255),2)
            videoWriter.write(current_img)
        del videoWriter
        
        
        
        
    def F_video(self):#write F value video
        
        file_name = 'F_value_video.avi'
        file_matrix = self.frame_sets
        self.video_plot(file_name,file_matrix)
        
        
    def dF_video(self):# Write dF value video
        
        self.dF_Matrix = np.zeros(shape = (self.frame_Num,472,472),dtype = np.float64)
        self.dF_Matrix_Original = np.zeros(shape = (self.frame_Num,472,472),dtype = np.float64)
        for i in range(self.frame_Num):#dF_Matrix Calculation,clip added
            current_graph = self.frame_sets[i,:,:]-self.average_graph
            bottom = current_graph.mean()-2*current_graph.std()
            top = current_graph.mean()+2*current_graph.std()        
            self.dF_Matrix_Original[i,:,:] = current_graph
            self.dF_Matrix[i,:,:] = np.clip(current_graph,bottom,top)
            
        self.dF_Matrix = (self.dF_Matrix-self.dF_Matrix.min())/(self.dF_Matrix.max()-self.dF_Matrix.min())
        self.dF_Matrix = np.uint8(self.dF_Matrix*255)
        file_name = 'dF_video.avi'
        self.video_plot(file_name,self.dF_Matrix)
        
    def df_F_video(self):#This function Generate dF/F graph, making it more easily being understood
        
        self.sub_Matrix = self.dF_Matrix_Original/self.average_graph
        for i in range(self.frame_Num):
            current_graph = self.sub_Matrix[i,:,:]
            bottom = current_graph.mean()-2*current_graph.std()
            top = current_graph.mean()+2*current_graph.std()
            self.sub_Matrix[i,:,:] = np.clip(current_graph,bottom,top)
        self.sub_Matrix = (self.sub_Matrix-self.sub_Matrix.min())/(self.sub_Matrix.max()-self.sub_Matrix.min())
        self.sub_Matrix = np.uint8(self.sub_Matrix*255)
        file_name = 'sub_video.avi'
        self.video_plot(file_name,self.sub_Matrix)
            
            
            
        
        
        
if __name__ == '__main__':
    
    filt = True# Whether filter is used to smoothing videos
    aligned_tif_folder = r'E:\ZR\Data_Temp\190412_L74_LM\1-002\results\Aligned_Frames'
    start_frame = 0
    stop_frame = 2220
    show_gain = 32#Gain,32 for GA mode.
    VW = Video_Write(aligned_tif_folder,start_frame,stop_frame,show_gain,filt)
    VW.read_in()
    VW.F_video()
    #VW.dF_video()
    #VW.df_F_video()
    test = VW.all_tif_name
    test_b = VW.frame_sets