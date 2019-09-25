# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:15:27 2019

@author: ZR
This tool will shuffle spike_train data, including switch cell data in a single frame,
And switch time data in a single cell.
"""

import General_Functions.my_tools as pp
import numpy as np
import random

class Data_Shuffle(object):
    
    name = r'Shuffle datas'
    
    def __init__(self,save_folder,spike_train_name,shuffle_times,shuffle_type):
        
        self.save_folder = save_folder
        self.spike_train = pp.read_variable(save_folder+r'\\'+spike_train_name)
        self.N_cell,self.N_frame = np.shape(self.spike_train)
        self.shuffled_folder = save_folder+r'\\Shuffled_trains'
        pp.mkdir(self.shuffled_folder)
        self.shuffle_times = shuffle_times #How many times you want to shuffle
        self.shuffle_type = shuffle_type####in_frame or cross_frames
    
        
        
    def shuffle_in_frame(self,shuffle_count):#This part will shuffle cell in every frame, with frame series not change
        
        shuffled_data_in_frame = np.zeros(shape = (self.N_cell,self.N_frame),dtype = np.float64)
        for i in range(self.N_frame):
            shuffled_data_in_frame[:,i] = random.sample(list(self.spike_train[:,i]),self.N_cell)
        pp.save_variable(shuffled_data_in_frame,self.shuffled_folder+r'\\spike_train_shuffle_in_frame_'+str(shuffle_count)+'.pkl')
        
    def shuffle_cross_frames(self,shuffle_count):#This part will shuffle every spike train across frame, with cell location not change
        
        shuffled_data_cross_frames = np.zeros(shape = (self.N_cell,self.N_frame),dtype = np.float64)
        for i in range(self.N_cell):
            shuffled_data_cross_frames[i,:] = random.sample(list(self.spike_train[i,:]),self.N_frame)
        pp.save_variable(shuffled_data_cross_frames,self.shuffled_folder+r'\\spike_train_shuffle_cross_frames_'+str(shuffle_count)+'.pkl')
    
    def main(self):
        
        for i in range(self.shuffle_times):
            if self.shuffle_type == 'in':
                self.shuffle_in_frame(i)
            else:
                self.shuffle_cross_frames(i)
                
    
    
if __name__ == '__main__':
    
    save_folder = r'E:\ZR\Data_Temp\190412_L74_LM\1-002\results'
    spike_train_name = 'spike_train_Morphology_filtered.pkl'
    shuffle_times = 500
    shuffle_type = 'cross'#############'in' means in frame shuffle, 'cross' means cross frames shuffle
    DS = Data_Shuffle(save_folder,spike_train_name,shuffle_times,shuffle_type)
    DS.main()
#    DS.shuffle_in_frame()
#    DS.shuffle_cross_frames()
#####################Use this function to generate 500 shuffled frames
