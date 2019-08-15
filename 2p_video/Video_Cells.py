# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:10:48 2019

@author: ZR
This Function will write videos using cell data
Only cell are plotted in this function in order to be more significant.
F graph plot
dF graph plot
dF/F graph plot
"""


import cv2
import numpy as np
import General_Functions.my_tools as pp

class Cell_Video(object):
    
    name = r'Generating cell videos of data series'
    
    def __init__(self,cell_group,save_folder,show_gain,start_frame,stop_frame):
        
        self.cell_group = cell_group
        self.save_folder = save_folder
        self.show_gain = show_gain
        self.frame_Num = stop_frame-start_frame
        
    def frame_sets_generation(self):
        
        self.frame_sets = np.zeros(shape = (self.frame_Num,512,512),dtype = np.float64)        
        aligned_tif_folder = save_folder+r'\\Aligned_Frames'
        all_tif_name = pp.file_name(aligned_tif_folder,'.tif')
        selected_frame_name = all_tif_name[start_frame:stop_frame]
        for i in range(self.frame_Num):
            temp_tif = cv2.imread(selected_frame_name[i],-1)
            self.frame_sets[i,:,:] = show_gain*temp_tif/256
            
    def F_calculation(self):# Calculate F value of every cell, in order to produce F series
        
        ###############Calculate F first
        self.F_series = np.zeros(shape = (self.frame_Num,len(self.cell_group)),dtype = np.float64)
        for i in range(self.frame_Num):#Cycle all cells
            for j in range(len(self.cell_group)):#Cycle all frames
                self.F_series[i,j] = pp.sum_a_frame(self.frame_sets[i,:,:],self.cell_group[j])
        ###################Then calculate dF/F
        self.sub_series = np.zeros(shape = (self.frame_Num,len(self.cell_group)),dtype = np.float64)
        for i in range(self.frame_Num):
            for j in range(len(self.cell_group)):
                cell_average = self.F_series[:,j].mean()
                self.sub_series[i,j] = (self.F_series[i,j]-cell_average)/cell_average
        
    def video_plot(self,file_name,file_matrix):#Give in adjusted matrix, return video 
        
        fps = 4
        size = (512,512)
        videoWriter = cv2.VideoWriter(save_folder+r'\\'+file_name,-1,fps,size,0)#last variance: is color
        for i in range(self.frame_Num):
            current_img = file_matrix[i,:,:]
            ##########Test Context below   
            cv2.putText(current_img,'Frame'+str(start_frame+i),(10,30),cv2.FONT_HERSHEY_PLAIN,2,(255),2)
            videoWriter.write(current_img)
        del videoWriter
        
    def F_video(self):#F_video is 0-1 video on black back ground, show avtivations.
        
        #clip F_series to +-2std
        self.F_series = np.clip(self.F_series,self.F_series.mean()-2*self.F_series.std(),self.F_series.mean()+2*self.F_series.std())
        # Then normalize F series to (0,1)
        self.F_series = (self.F_series-self.F_series.min())/(self.F_series.max()-self.F_series.min())
        #recover cell data to graph datas.
        F_video_context = np.zeros(shape = (self.frame_Num,512,512),dtype = np.uint8)
        for i in range(self.frame_Num):
            current_frame = np.zeros(shape = (512,512),dtype = np.uint8)
            for j in range(len(self.cell_group)):
                x_list,y_list = pp.cell_location(self.cell_group[j])
                current_frame[y_list,x_list] = np.uint8(self.F_series[i,j]*255)
            F_video_context[i,:,:] = current_frame
        self.video_plot('F_Video_Cell.avi',F_video_context)
    
    def sub_video(self):#dF video is based on 0.5 graph, use 0 as gray.
        
        #Clip First, use +-4std as standart to keep most data unchanged.
        self.sub_series = np.clip(self.sub_series,self.sub_series.mean()-2*self.sub_series.std(),self.sub_series.mean()+2*self.sub_series.std())
        #Then normalize all data,keep 0 as 0 unchanged.
        self.sub_series = self.sub_series/abs(self.sub_series).max()
        #Recover cell data to frame
        sub_video_context = np.zeros(shape = (self.frame_Num,512,512),dtype = np.uint8)
        for i in range(self.frame_Num):
            current_frame = np.ones(shape = (512,512),dtype = np.uint8)*127
            for j in range(len(self.cell_group)):
                x_list,y_list = pp.cell_location(self.cell_group[j])
                current_frame[y_list,x_list] = np.uint8((self.sub_series[i,j]+1)*127)
            sub_video_context[i,:,:] = current_frame
        self.video_plot('Sub_Video_Cell.avi',sub_video_context)
    
    
if __name__ == '__main__':
    
    save_folder = r'E:\ZR\Data_Temp\190412_L74_LM\1-002\results'
    cell_group = pp.read_variable(save_folder+r'\\Cell_Groups_Morphology.pkl')
    show_gain = 32
    start_frame = 1000
    stop_frame = 1500
    CV = Cell_Video(cell_group,save_folder,show_gain,start_frame,stop_frame)
    CV.frame_sets_generation()
    CV.F_calculation()
    #CV.F_video()
    CV.sub_video()
    b = CV.F_series
    